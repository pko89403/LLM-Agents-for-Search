import logging
from typing import Optional

from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import END, StateGraph

from llm_utils import get_default_llm
from nodes import (
    node_decide,
    node_finish,
    node_lookup,
    node_retrieve,
    node_search,
)
from state import AgentState


def build_knowagent_graph(
    llm: Optional[BaseLanguageModel] = None,
    max_consec_search: int = 3,
    auto_finish_step: int = 6,
    context_len: int = 2000,
):
    """KnowAgent용 LangGraph 그래프를 구성/컴파일하여 반환합니다."""
    logging.info("LangGraph 그래프 빌드 시작...")
    if StateGraph is None:
        raise RuntimeError(
            "langgraph가 설치되어 있지 않습니다. `pip install langgraph`를 실행하세요."
        )

    # decide 노드에서 사용할 LLM 캡처
    if llm is None:
        llm = get_default_llm()

    def decide_wrapper(state: AgentState) -> AgentState:
        logging.info(f"Step {state.get('step', 1)}: decide 노드 실행 중...")
        return node_decide(state, llm=llm, max_consec_search=max_consec_search, auto_finish_step=auto_finish_step, context_len=context_len)  # type: ignore

    graph = StateGraph(AgentState)

    graph.add_node("decide", decide_wrapper)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("search", node_search)
    graph.add_node("lookup", node_lookup)
    graph.add_node("finish", node_finish)

    graph.set_entry_point("decide")

    # decide에서 다음 노드를 분기
    def route_from_decide(state: AgentState):
        action_type = (state.get("action_type") or "").title()
        logging.info(
            f"Step {state.get('step', 1)}: decide 노드 완료. 다음 액션: {action_type}"
        )
        if action_type == "Retrieve":
            return "retrieve"
        if action_type == "Search":
            return "search"
        if action_type == "Lookup":
            return "lookup"
        if action_type == "Finish":
            return "finish"
        # 폴백: 복구 시도용 search
        logging.warning(
            f"알 수 없는 액션 타입 '{action_type}' 감지. Search로 폴백합니다."
        )
        return "search"

    graph.add_conditional_edges(
        "decide",
        route_from_decide,
        {  # type: ignore
            "retrieve": "retrieve",
            "search": "search",
            "lookup": "lookup",
            "finish": "finish",
        },
    )

    # 각 도구 실행 후, 종료되지 않았다면 다시 decide로
    def continue_or_end(state: AgentState):
        current_node = state.get("__previous_node__", "Unknown")
        if state.get("finished"):
            logging.info(
                f"Step {state.get('step', 1)}: {current_node} 노드 완료. 에이전트 종료."
            )
            return END
        else:
            logging.info(
                f"Step {state.get('step', 1)}: {current_node} 노드 완료. decide 노드로 돌아갑니다."
            )
            return "decide"

    graph.add_conditional_edges("retrieve", continue_or_end, {"decide": "decide", END: END})  # type: ignore
    graph.add_conditional_edges("search", continue_or_end, {"decide": "decide", END: END})  # type: ignore
    graph.add_conditional_edges("lookup", continue_or_end, {"decide": "decide", END: END})  # type: ignore
    graph.add_edge("finish", END)

    app = graph.compile()
    logging.info("LangGraph 그래프 빌드 완료.")
    return app


def run_knowagent(
    question: str,
    llm: Optional[BaseLanguageModel] = None,
    max_steps: int = 12,
    max_consec_search: int = 3,
    auto_finish_step: int = 6,
    context_len: int = 2000,
) -> AgentState:
    """질문을 입력받아 그래프를 실행하고 최종 상태를 반환합니다."""
    logging.info(f"KnowAgent 실행 시작 (최대 {max_steps} 스텝)...")
    app = build_knowagent_graph(
        llm,
        max_consec_search=max_consec_search,
        auto_finish_step=auto_finish_step,
        context_len=context_len,
    )

    state: AgentState = {
        "question": question,
        "scratchpad": "",
        "step": 1,
        "last_passages": [],
        "finished": False,
    }

    # app.invoke를 사용하여 단일 호출로 실행 (recursion_limit로 최대 스텝 제어)
    final_state: AgentState = app.invoke(state, config={"recursion_limit": max_steps})
    logging.info("KnowAgent 실행 완료.")
    logging.debug(f"Final state before return: {final_state}")  # Added debug log
    return final_state
