"""
에이전트의 핵심 의사결정 및 도구 실행 노드 구현
"""

from typing import Dict, Any
from agentq.state import AgentState, increment_loop_count, add_error, clear_error
from agentq.llm_utils import get_llm_manager
from agentq.prompt_utils import (
    get_prompt_builder, ScratchpadManager,
    extract_action_from_response, extract_critique_decision, clean_response
)
from agentq.tools import get_tool_executor


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
    """Thought 노드: 다음 행동 결정"""
    print(f"🤔 Thought 노드 실행 중... (루프 {state['loop_count'] + 1})")
    try:
        # 루프 카운트 증가
        state = increment_loop_count(state)

        # 프롬프트 구성
        prompt_builder = get_prompt_builder()
        system_prompt = prompt_builder.build_thought_prompt(state)

        # LLM 호출
        llm_manager = get_llm_manager()
        response = await llm_manager.invoke_with_system(
            system_prompt=system_prompt,
            user_message="What should be the next action based on the current situation?"
        )

        # 응답 정리 및 액션 추출
        thought = clean_response(response)
        action = extract_action_from_response(response)

        # 상태 업데이트
        state["thought"] = thought
        state["action"] = action

        # 스크래치패드에 추가
        state = ScratchpadManager.add_thought(state, thought)
        if action:
            state = ScratchpadManager.add_action(state, action)

        print(f"   사고 과정: {thought[:100]}...")
        if action:
            print(f"   계획된 액션: {action.get('type')}")
        else:
            print("   액션이 추출되지 않았습니다.")

        return {
            "thought": thought,
            "action": action,
            "loop_count": state["loop_count"]
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