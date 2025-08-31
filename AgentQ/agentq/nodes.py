"""
에이전트의 핵심 의사결정 및 도구 실행 노드 구현
"""

from typing import Dict, Any
from agentq.state import AgentState, increment_loop_count, add_error, clear_error
from agentq.llm_utils import get_llm_manager
from agentq.prompt_utils import (
    get_prompt_builder, ScratchpadManager,
    extract_action_from_response, extract_critique_decision, clean_response,
    split_output_blocks, extract_commands_and_status, parse_command_line
)
from agentq.tools import get_tool_executor, WebTool
import re
import json

# 릴리즈 토글/가드
ENABLE_CRITIC = True
ENABLE_MCTS_LITE = True


async def plan_node(state: AgentState) -> Dict[str, Any]:
    """Plan 노드: 전체 계획 수립"""
    print("📋 Plan 노드 실행 중...")

    try:
        # 프롬프트 구성
        prompt_builder = get_prompt_builder()
        system_prompt = prompt_builder.build_plan_prompt(state)

        # LLM 호출
        llm_manager = get_llm_manager()
        response = await llm_manager.invoke_with_system(
            system_prompt=system_prompt,
            user_message=f"Please create a plan for: {state['objective']}"
        )

        # 응답 정리
        plan = clean_response(response)

        # 상태 업데이트
        state["plan"] = plan

        # 스크래치패드에 추가
        state = ScratchpadManager.add_plan(state, plan)

        print(f"   계획 수립 완료: {plan[:100]}...")
        return {"plan": plan}

    except Exception as e:
        error_msg = f"Plan 노드 실행 중 오류: {str(e)}"
        print(f"❌ {error_msg}")
        state = add_error(state, error_msg)
        return {"plan": "계획 수립에 실패했습니다."}


async def thought_node(state: AgentState) -> Dict[str, Any]:
    """Thought 노드: 후보 생성 → critic 점수화 → 최종 액션 선택"""
    print(f"🤔 Thought 노드 실행 중... (루프 {state['loop_count'] + 1})")
    
    try:
        # 루프 카운트 증가
        state = increment_loop_count(state)

        # 최신 DOM 스냅샷 확보 (없을 때만)
        if not state.get("page_content"):
            snap = await WebTool.extract_page_content()
            if snap.get("success") and isinstance(snap.get("data"), dict):
                d = snap["data"]
                state["current_url"] = d.get("url")
                state["page_title"] = d.get("title")
                state["page_content"] = (d.get("content") or "")[:500]
                state["observation"] = "페이지 내용 추출 완료"

        prompt_builder = get_prompt_builder()
        llm_manager = get_llm_manager()
        
        action = None
        thought_text = ""
        cmds = []
        status = "CONTINUE"
        best_cmd = None
        scores = []

        if ENABLE_CRITIC:
            # 후보 커맨드 생성 (복잡한 프롬프트 사용)
            system_prompt = prompt_builder.build_thought_prompt(state)
            resp = await llm_manager.invoke_with_system(
                system_prompt=system_prompt,
                user_message="Propose multiple candidate COMMANDS for the very next step."
            )
            raw = clean_response(resp)
            cmds, status = extract_commands_and_status(raw)
            thought_text = split_output_blocks(raw).get("THOUGHT", "")
            
            if cmds and ENABLE_MCTS_LITE:
                try:
                    # critic으로 랭킹
                    critic_prompt = prompt_builder.build_critic_prompt(state)
                    critic_in = "Rank these commands:\n" + "\n".join([f"{i+1}. {c}" for i,c in enumerate(cmds)])
                    critic_out = await llm_manager.invoke_with_system(critic_prompt, critic_in)
                    
                    # LLM이 생성한 JSON 앞뒤의 불필요한 텍스트(예: ```json ... ```)를 제거합니다.
                    match = re.search(r'\[.*?\]', critic_out, re.DOTALL)
                    json_str = match.group(0) if match else critic_out
                    
                    parsed = json.loads(json_str)
                    
                    if isinstance(parsed, list):
                        for p in parsed:
                            if isinstance(p, dict):
                                # 'cmd' 또는 '"cmd"'와 같이 따옴표가 포함된 키도 처리합니다.
                                cmd_key = next((k for k in p if 'cmd' in k), None)
                                score_key = next((k for k in p if 'score' in k), None)
                                
                                if cmd_key and score_key:
                                    cmd = p[cmd_key]
                                    score = float(p[score_key])
                                    scores.append((cmd, score))

                except Exception as e:
                    print(f"--- Critic 파싱 실패: {e} ---")
                    print(f"--- Critic 원본 출력 ---\n{critic_out}\n--------------------")
                    # 파싱 실패 시, 후보들에게 균등한 점수를 부여하여 로직을 계속 진행합니다.
                    if not scores and cmds:
                        scores = [(c, 0.5) for c in cmds]
            
            # critic 출력이 불완전하면 균등 분배
            if not scores and cmds:
                scores = [(c, 0.5) for c in cmds]

            if scores:
                # Q 통계와 결합 (간단한 UCB-lite)
                qstats = state.get("q_stats") or {}
                alpha = 0.5
                scored = []
                for c, s in scores:
                    q = qstats.get(c, {}).get("Q", 0.0)
                    n = qstats.get(c, {}).get("N", 0)
                    bonus = 0.1 if n == 0 else 0.0
                    total = alpha * s + (1 - alpha) * q + bonus
                    scored.append((total, c))
                scored.sort(reverse=True)
                best_cmd = scored[0][1]
                action = parse_command_line(best_cmd)
            else:
                # Critic/MCTS-lite 실패 시 Option A (단일 액션)으로 폴백
                print("⚠️ Critic/MCTS-lite 실패 또는 후보 없음. Option A로 폴백합니다.")
                # Fallback to Option A logic
                system_prompt = prompt_builder.build_thought_prompt(state) # 이 프롬프트는 이제 단순화된 버전
                resp = await llm_manager.invoke_with_system(
                    system_prompt=system_prompt,
                    user_message="What should be the next action based on the current situation?"
                )
                thought_text = clean_response(resp)
                action = extract_action_from_response(thought_text) # 단순화된 파서 사용
        else: # ENABLE_CRITIC is False
            # Option A (단일 액션) 프롬프트 사용
            system_prompt = prompt_builder.build_thought_prompt(state) # 이 프롬프트는 이제 단순화된 버전
            resp = await llm_manager.invoke_with_system(
                system_prompt=system_prompt,
                user_message="What should be the next action based on the current situation?"
            )
            thought_text = clean_response(resp)
            action = extract_action_from_response(thought_text) # 단순화된 파서 사용

        # 상태 업데이트
        state["thought"] = thought_text or "다음 행동을 결정했습니다."
        state["candidate_commands"] = cmds
        state["critic_scores"] = [s for _, s in scores] if scores else []
        state["status"] = status
        state["last_command"] = best_cmd if 'best_cmd' in locals() else None
        state["action"] = action

        # 스크래치패드 기록
        state = ScratchpadManager.add_thought(state, state["thought"])
        if action:
            state = ScratchpadManager.add_action(state, action)

        print(f"   사고 과정: {state["thought"][:100]}...")
        if action:
            print(f"   계획된 액션: {action.get('type')}")
        else:
            print("   액션이 추출되지 않았습니다.")
        
        return {
            "thought": state["thought"],
            "action": action,
            "loop_count": state["loop_count"],
            "candidate_commands": cmds
        }
    except Exception as e:
        error_msg = f"Thought 노드 실행 중 오류: {str(e)}"
        print(f"❌ {error_msg}")
        state = add_error(state, error_msg)
        return {
            "thought": "다음 행동을 결정하는데 실패했습니다.",
            "action": None,
            "loop_count": state["loop_count"]
        }


async def action_node(state: AgentState) -> Dict[str, Any]:
    """Action 노드: 실제 액션 실행"""
    print("⚡ Action 노드 실행 중...")

    try:
        action = state.get("action")

        if not action:
            observation = "실행할 액션이 없습니다."
            print(f"   {observation}")
            state["observation"] = observation
            state = ScratchpadManager.add_observation(state, observation)
            return {"observation": observation}

        # 도구 실행
        tool_executor = get_tool_executor()
        result = await tool_executor.execute_action(action)

        # 관찰 결과 구성
        if result["success"]:
            observation = result["message"]
            if result.get("data"):
                if isinstance(result["data"], dict):
                    # 페이지 정보 업데이트
                    if "url" in result["data"]:
                        state["current_url"] = result["data"]["url"]
                    if "title" in result["data"]:
                        state["page_title"] = result["data"]["title"]
                    if "content" in result["data"]:
                        state["page_content"] = result["data"]["content"][:500]  # 처음 500자만
                        observation += f"\n페이지 내용: {result['data']['content'][:200]}..."
                elif isinstance(result["data"], str):
                    observation += f"\n결과: {result['data'][:200]}..."
        else:
            observation = f"액션 실패: {result['message']}"

        # 상태 업데이트
        state["observation"] = observation
        state = ScratchpadManager.add_observation(state, observation)

        # 에러 상태 초기화 (성공한 경우)
        if result["success"]:
            state = clear_error(state)

        print(f"   실행 결과: {observation[:100]}...")
        return {"observation": observation}

    except Exception as e:
        error_msg = f"Action 노드 실행 중 오류: {str(e)}"
        print(f"❌ {error_msg}")
        observation = f"액션 실행 중 오류가 발생했습니다: {str(e)}"
        state["observation"] = observation
        state = add_error(state, error_msg)
        state = ScratchpadManager.add_observation(state, observation)
        return {"observation": observation}


async def explanation_node(state: AgentState) -> Dict[str, Any]:
    """Explanation 노드: 결과 설명"""
    print("📝 Explanation 노드 실행 중...")

    try:
        # 프롬프트 구성
        prompt_builder = get_prompt_builder()
        system_prompt = prompt_builder.build_explanation_prompt(state)

        # LLM 호출
        llm_manager = get_llm_manager()
        response = await llm_manager.invoke_with_system(
            system_prompt=system_prompt,
            user_message="Please explain what happened and its significance."
        )

        # 응답 정리
        explanation = clean_response(response)

        # 상태 업데이트
        state["explanation"] = explanation
        state = ScratchpadManager.add_explanation(state, explanation)

        print(f"   설명: {explanation[:100]}...")
        return {"explanation": explanation}

    except Exception as e:
        error_msg = f"Explanation 노드 실행 중 오류: {str(e)}"
        print(f"❌ {error_msg}")
        explanation = f"결과 해석 중 오류가 발생했습니다: {str(e)}"
        state["explanation"] = explanation
        state = add_error(state, error_msg)
        return {"explanation": explanation}


async def critique_node(state: AgentState) -> Dict[str, Any]:
    """Critique 노드: 완료 여부 판단"""
    print("🔍 Critique 노드 실행 중...")

    try:
        # 프롬프트 구성
        prompt_builder = get_prompt_builder()
        system_prompt = prompt_builder.build_critique_prompt(state)

        # LLM 호출
        llm_manager = get_llm_manager()
        response = await llm_manager.invoke_with_system(
            system_prompt=system_prompt,
            user_message="Should we continue or is the task complete?"
        )

        # 응답 정리
        critique = clean_response(response)

        # 완료 여부 결정
        done = extract_critique_decision(response)

        # 도메인 휴리스틱 (OpenTable)
        try:
            url = state.get("current_url") or ""
            text = (state.get("page_content") or "").lower()
            if "opentable.com" in url:
                success_keywords = ["reservation confirmed", "complete reservation", "you're all set"]
                if any(k in text for k in success_keywords):
                    done = True
                    critique += "\nHeuristic: OpenTable success indicators found."
        except Exception:
            pass

        # 간단한 Q-통계 업데이트 (세션 내에서만)
        if ENABLE_MCTS_LITE:
            try:
                cmd = state.get("last_command")
                if cmd:
                    stats = state.get("q_stats") or {}
                    ent = stats.get(cmd, {"Q": 0.0, "N": 0})
                    reward = 1.0 if done else 0.0
                    ent["N"] += 1
                    ent["Q"] = ent["Q"] + (reward - ent["Q"]) / ent["N"]
                    stats[cmd] = ent
                    state["q_stats"] = stats
            except Exception:
                pass

        # 최대 루프 횟수 체크
        if state["loop_count"] >= state["max_loops"]:
            done = True
            critique += f"\n최대 루프 횟수({state['max_loops']})에 도달하여 종료합니다."

        # 상태 업데이트
        state["done"] = done
        state = ScratchpadManager.add_critique(state, critique, done)

        status = "완료" if done else "계속"
        print(f"   평가 결과: {status} - {critique[:100]}...")

        return {
            "done": done,
            "critique": critique
        }

    except Exception as e:
        error_msg = f"Critique 노드 실행 중 오류: {str(e)}"
        print(f"❌ {error_msg}")

        # 오류 발생 시 안전하게 종료
        critique = f"평가 중 오류가 발생했습니다: {str(e)}"
        state["done"] = True
        state = add_error(state, error_msg)

        return {
            "done": True,
            "critique": critique
        }


# 라우팅 함수
def should_continue(state: AgentState) -> str:
    """다음 노드 결정"""
    if state.get("done", False):
        return "end"
    else:
        return "thought"


def check_max_loops(state: AgentState) -> str:
    """최대 루프 횟수 체크"""
    if state["loop_count"] >= state["max_loops"]:
        return "critique"  # 강제로 critique로 보내서 종료 처리
    return "action"