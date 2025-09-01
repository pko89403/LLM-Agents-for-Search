"""
에이전트의 핵심 의사결정 및 도구 실행 노드 구현
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field
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




def _progress_fingerprint(state: AgentState) -> str:
    from hashlib import md5
    url = (state.get("current_url") or "")
    title = (state.get("page_title") or "")
    content = (state.get("page_content") or "")[:1000]
    return md5(f"{url}|{title}|{content}".encode("utf-8")).hexdigest()


# --- Pydantic 모델 정의 ---
from pydantic import BaseModel, Field
from typing import List, Dict, Any


class ThoughtProcess(BaseModel):
    """LLM의 사고 과정을 담는 구조화된 모델"""
    plan: str = Field(description="The overall plan to achieve the objective.")
    thought: str = Field(description="Concise reasoning for the next action candidates.")
    commands: List[str] = Field(description="A list of 3-5 candidate commands for the next step.")
    status: str = Field(description="Should be 'CONTINUE' if the task is not yet complete.")

class CriticScore(BaseModel):
    """Critic이 평가한 개별 명령어 점수"""
    cmd: str = Field(description="The verbatim command being scored.")
    score: float = Field(description="The score from 0.0 to 1.0.")
    rationale: str = Field(description="A short rationale for the score.")

class CriticOutput(BaseModel):
    """Critic의 전체 평가 결과"""
    scores: List[CriticScore]


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
    """Thought 노드: 후보 생성 → critic 점수화 → 최종 액션 선택 (간소화된 버전)"""
    print(f"🤔 Thought 노드 실행 중... (루프 {state['loop_count'] + 1})")

    try:
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
        
        # 1. 후보 커맨드 생성
        thought_prompt = prompt_builder.build_thought_prompt(state)
        thought_process: ThoughtProcess = await llm_manager.invoke_structured_with_system(
            system_prompt=thought_prompt,
            user_message="Propose multiple candidate COMMANDS for the very next step.",
            schema_model=ThoughtProcess
        )

        if not thought_process or not thought_process.commands:
            raise ValueError("LLM으로부터 유효한 커맨드를 생성하지 못했습니다.")

        cmds = thought_process.commands
        thought_text = thought_process.thought
        status = thought_process.status

        scores = []
        # 2. Critic으로 랭킹 (활성화된 경우)
        if ENABLE_CRITIC and cmds:
            critic_prompt = prompt_builder.build_critic_prompt(state)
            critic_input = "Rank these commands:\n" + "\n".join([f"- {c}" for c in cmds])
            
            critic_output: CriticOutput = await llm_manager.invoke_structured_with_system(
                system_prompt=critic_prompt,
                user_message=critic_input,
                schema_model=CriticOutput
            )

            if critic_output and critic_output.scores:
                score_map = {item.cmd.strip(): item.score for item in critic_output.scores}
                scores = [(cmd, score_map.get(cmd.strip(), 0.5)) for cmd in cmds]
                print(f"   Critic 점수: { {c: s for c, s in scores} }")

        # 점수화 실패 시 균등 분배
        if not scores and cmds:
            scores = [(c, 0.5) for c in cmds]

        # 3. 최종 액션 선택 (MCTS-lite)
        best_cmd = ""
        action = None
        if scores:
            if ENABLE_MCTS_LITE:
                qstats = state.get("q_stats") or {}
                alpha = 0.5
                scored_cmds = []
                for c, s in scores:
                    q = qstats.get(c, {}).get("Q", 0.0)
                    n = qstats.get(c, {}).get("N", 0)
                    bonus = 0.1 if n == 0 else 0.0
                    total = alpha * s + (1 - alpha) * q + bonus
                    scored_cmds.append((total, c))
                
                scored_cmds.sort(reverse=True)
                best_cmd = scored_cmds[0][1]
                print(f"   UCB-lite 상위: { {c: t for t, c in scored_cmds[:3]} }")
            else:
                # MCTS 비활성화 시 critic 점수만으로 선택
                best_cmd = max(scores, key=lambda item: item[1])[0]
            
            action = parse_command_line(best_cmd)

        # Fallback: 액션 선택 실패 시
        if not action:
            best_cmd = cmds[0] if cmds else "GET_DOM"
            action = parse_command_line(best_cmd)

        # 4. 상태 업데이트
        state.update({
            "thought": thought_text,
            "candidate_commands": cmds,
            "critic_scores": [s for _, s in scores],
            "status": status,
            "last_command": best_cmd,
            "action": action,
        })

        state = ScratchpadManager.add_thought(state, state["thought"])
        if action:
            state = ScratchpadManager.add_action(state, action)

        print(f"   사고 과정: {state['thought'][:100]}...")
        print(f"   계획된 액션: {action.get('type') if action else '없음'}")

        return {
            "thought": state["thought"],
            "action": action,
            "loop_count": state["loop_count"],
            "candidate_commands": cmds
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = f"Thought 노드 실행 중 오류: {str(e)}"
        print(f"❌ {error_msg}")
        state = add_error(state, error_msg)
        return {
            "thought": "다음 행동을 결정하는데 실패했습니다.",
            "action": {"type": "GET_DOM"}, # 에러 발생 시 안전한 액션으로 폴백
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

        # ---- 진행도/루프 가드 ----
        loops = state.get("loop_count", 0)
        min_loops = state.get("min_loops", 3)

        prev_fp = state.get("last_progress_fingerprint")
        curr_fp = _progress_fingerprint(state)
        progress_gain = (prev_fp != curr_fp) and bool(curr_fp)

        if progress_gain:
            state["no_progress_streak"] = 0
            state["last_progress_fingerprint"] = curr_fp
        else:
            state["no_progress_streak"] = state.get("no_progress_streak", 0) + 1

        if loops < min_loops:
            done = False

        if state.get("no_progress_streak", 0) >= 3 and loops >= min_loops:
            done = True
            critique += "\nHeuristic: No progress for multiple steps → stopping."

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
                    reward = 1.0 if done else (0.2 if progress_gain else 0.0)
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

        print(f"   루프 {loops}, min_loops {min_loops}, no_progress_streak {state.get('no_progress_streak')}, done={done}")

        # 상태 업데이트
        state["done"] = done
        state = ScratchpadManager.add_critique(state, critique, done)

        status = "완료" if done else "계속"
        print(f"   평가 결과: {status} - {critique[:100]}...")

        return {
            "done": done,
            "critique": critique,
            "no_progress_streak": state["no_progress_streak"],
            "last_progress_fingerprint": state.get("last_progress_fingerprint")
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
