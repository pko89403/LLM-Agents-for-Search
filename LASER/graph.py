# -*- coding: utf-8 -*-
"""에이전트의 상태 그래프를 정의하고 빌드합니다."""

import logging
from typing import Optional, Any
from functools import partial

from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import LaserState
from nodes import node_search_space, node_result_space, node_item_space, node_stopping_space
from llm_utils import get_default_llm # 추가: LLM이 None일 경우 기본 LLM을 가져오기 위함

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def build_laser_graph(
    llm: Optional[BaseLanguageModel] = None,
    max_steps: int = 15,
    enable_feedback: bool = False
):
    """LASER 에이전트의 상태 그래프를 구성하고 컴파일하여 반환합니다."""

    # llm이 None이면 기본 LLM을 가져옵니다.
    if llm is None:
        llm = get_default_llm()

    # --- 클로저(Closure)로 라우터 함수 정의 ---
    def router_fn(state: LaserState) -> str:
        """상태에 따라 다음 노드를 결정하는 라우팅 함수입니다."""
        step = state.get('step_count', 0)
        logging.info(f"Step {step}: 라우팅 결정...")

        # 클로저를 통해 외부 함수(build_laser_graph)의 max_steps 변수에 직접 접근합니다.
        if step >= max_steps:
            logging.warning(f"최대 스텝({max_steps}) 도달. Stopping 노드로 강제 이동합니다.")
            return "to_stop"

        route = state.get("route")
        if not route:
            logging.error("'route'가 상태에 설정되지 않아 라우팅 불가.")
            raise ValueError("'route'가 상태에 설정되지 않았습니다.")

        logging.info(f"라우팅: state.route='{route}' -> 다음 노드로 이동")
        return route
    # -------------------------------------

    logging.info("LASER 그래프 빌드 시작...")

    graph = StateGraph(LaserState)

    # 각 상태에 해당하는 노드를 그래프에 추가합니다.
    graph.add_node("Search", partial(node_search_space, llm=llm, enable_feedback=enable_feedback))
    graph.add_node("Result", partial(node_result_space, llm=llm, enable_feedback=enable_feedback))
    graph.add_node("Item", partial(node_item_space, llm=llm, enable_feedback=enable_feedback))
    graph.add_node("Stopping", node_stopping_space)

    # 에이전트의 시작점은 Search 노드입니다.
    graph.set_entry_point("Search")

    # Search 노드는 항상 Result 노드로 이동합니다.
    graph.add_edge("Search", "Result")

    # Result 노드에서는 조건부로 다음 노드를 결정합니다.
    graph.add_conditional_edges(
        "Result",
        router_fn,
        {
            "to_item": "Item",
            "stay_result": "Result",
            "to_search": "Search",
            "to_stop": "Stopping",
        },
    )

    # Item 노드에서도 조건부로 다음 노드를 결정합니다.
    graph.add_conditional_edges(
        "Item",
        router_fn,
        {
            "stay_item": "Item",
            "to_result": "Result",
            "to_stop": "Stopping",
        },
    )

    # Stopping 노드는 그래프의 끝(END)으로 연결됩니다.
    graph.add_edge("Stopping", END)

    # 체크포인터(메모리)를 설정하여 상태를 기록할 수 있습니다.
    memory = MemorySaver()

    # 그래프를 컴파일하여 실행 가능한 app 객체로 만듭니다.
    app = graph.compile()
    logging.info("LASER 그래프 빌드 완료.")
    return app


def run_laser_agent(
    env: Any,
    instruction: str,
    initial_observation: str, # 초기 관찰을 직접 받도록 변경
    initial_url: Optional[str] = None, # 초기 URL을 직접 받도록 변경 (선택 사항)
    llm: Optional[BaseLanguageModel] = None,
    max_steps: int = 15,
    session_id: Optional[int] = None, # 추가: 세션 ID를 받도록 변경
    enable_feedback: bool = False, # 추가: 피드백 시스템 활성화 여부
) -> dict:
    """LASER 에이전트 그래프를 실행하고 최종 상태를 반환합니다.

    Args:
        env: .reset()과 .step() 메소드를 가진 환경 객체.
        instruction: 에이전트에게 전달할 초기 지시사항.
        initial_observation: 환경에서 얻은 초기 관찰.
        initial_url: 환경의 초기 URL (선택 사항).
        llm: 사용할 언어 모델.
        max_steps: 최대 실행 스텝 수.
        session_id: 현재 실행 중인 세션의 ID (체크포인터용).
    """
    logging.info(f"LASER 에이전트 실행 시작 (최대 {max_steps} 스텝)...")

    # 1. 그래프 빌드
    app = build_laser_graph(llm=llm, max_steps=max_steps, enable_feedback=enable_feedback)

    # 2. 초기 상태 정의
    initial_state: LaserState = {
        "user_instruction": instruction,
        "obs": initial_observation, # 직접 받은 초기 관찰 사용
        "url": initial_url, # 직접 받은 초기 URL 사용
        "current_laser_state": "Search",
        "step_count": 0,
        "action_history": [],
        "thought_history": [],
        "memory_buffer": [],
        "feedback_history": [], # 피드백 히스토리 추가
        "rethink_history": [], # 재고 히스토리 추가
        "_env": env, # 노드가 환경과 상호작용할 수 있도록 전달
    }

    # 3. 그래프 실행
    # recursion_limit으로 최대 스텝을 제어합니다.
    config = {"recursion_limit": max_steps * 2}
    if session_id is not None:
        config["configurable"] = {"thread_id": str(session_id)} # 체크포인터 설정

    final_state = app.invoke(initial_state, config=config)

    logging.info("LASER 에이전트 실행 완료.")
    return final_state
