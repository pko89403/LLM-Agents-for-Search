'''
ITTS(Inference-time Tree Search) 에이전트의 실행을 위한 LangGraph 그래프를 정의합니다.
'''
from langgraph.graph import StateGraph, END
from state import SearchState
from nodes import initialize_state, expand_frontier, check_finish_condition

def create_itts_agent_graph() -> StateGraph:
    """
    ITTS 에이전트 워크플로우를 정의하는 StateGraph를 생성합니다.
    그래프 구조: initialize -> expand -> (conditional) -> expand ... -> finish
    """
    workflow = StateGraph(SearchState)

    # 1. 노드 추가
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("expand_frontier", expand_frontier)

    # 2. 엣지(흐름) 정의
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "expand_frontier")

    # expand_frontier 노드 이후, 조건에 따라 분기
    workflow.add_conditional_edges(
        "expand_frontier",
        check_finish_condition,
        {
            "continue": "expand_frontier",  # 계속 탐색
            "finish": END,                 # 종료
        },
    )

    # 3. 그래프 컴파일
    return workflow.compile()

def run_agent(goal: str, max_steps: int = 10, branching: int = 5, budget: int = 64):
    """
    ITTS 에이전트를 실행합니다.
    """
    graph = create_itts_agent_graph()

    # 초기 상태 정의
    # SearchState TypedDict에 정의된 키만 사용해야 합니다.
    initial_state = {
        "goal": goal,
        "max_steps": max_steps,
        "branching": branching,
        "budget": budget,
        # 다른 키들은 노드 내에서 초기화되므로 여기서 정의하지 않습니다.
        # 예를 들어, frontier, search_counter 등은 initialize_state에서 설정됩니다.
    }

    print(f"목표: '{goal}', 최대 스텝: {max_steps}, 브랜칭 팩터: {branching}, 탐색 예산: {budget}")
    
    # 그래프 실행
    # config는 필요시 추가 (예: recursion_limit)
    final_state = graph.invoke(initial_state)

    # 최종 결과 출력
    print("\n--- 최종 결과 ---")
    best_state = final_state.get("best_state")
    if best_state:
        print(f"최고 점수: {final_state.get('best_score'):.4f}")
        print(f"최종 액션 이력: {best_state.get('action_history')}")
        print(f"최종 답변: {final_state.get('final_answer')}")
    else:
        print("유의미한 결과를 찾지 못했습니다.")

    return final_state
