# `langgraph-supervisor`와 전역 메시지 풀(블랙보드) 연동 예제

`langgraph-supervisor`를 전역 메시지 풀 스키마(`pool: Annotated[List[AnyMessage], add_messages]`)와 연동하여 실행 가능한 형태로 변형한 버전입니다. Supervisor는 대화를 관리하고, 그 결과를 전역 풀에 미러링하여 블랙보드(SSOT)에 축적하는 구조입니다.

## 설치
```bash
pip install -U langgraph-supervisor langgraph langchain-openai
export OPENAI_API_KEY=<YOUR_KEY>
```

## supervisor_with_global_pool.py
```python
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

# ---------------------------
# 1) Global State (전역 메시지 풀 스키마)
# ---------------------------
class GlobalState(TypedDict):
    pool: Annotated[List[AnyMessage], add_messages]  # 글로벌 메시지 허브(SSOT)
    profile: Dict[str, Any]
    metrics: Dict[str, Any]

def pool_append(content: str, **meta) -> Dict[str, Any]:
    """Supervisor 결과를 전역 풀에 append하기 위한 헬퍼."""
    msg = AIMessage(
        content=content,
        additional_kwargs={"metadata": meta}  # 예: stage/route/score/from/to/scope/topic/ts
    )
    return {"pool": [msg]}

def pool_view_for_supervisor(state: GlobalState) -> List[Dict[str, str]]:
    """
    Supervisor/에이전트에 전달할 messages 뷰 구성.
    - 블랙보드에 쌓인 Human/AI 메시지를 standard {role, content}로 변환.
    - 필요 시 scope/topic 필터링 로직을 추가 가능.
    """
    view: List[Dict[str, str]] = []
    for m in state["pool"]:
        role = "assistant"
        if isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, SystemMessage):
            role = "system"
        view.append({"role": role, "content": m.content})
    return view

# ---------------------------
# 2) LLM & 도메인별 에이전트 정의
# ---------------------------
model = ChatOpenAI(model="gpt-4o-mini")

# 간단한 툴들 (예시)
def add(a: float, b: float) -> float: return a + b
def multiply(a: float, b: float) -> float: return a * b
def web_search(query: str) -> str: return "Example search result for: " + query

math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    prompt="You are a math expert. Use tools as needed; one tool call at a time."
)

research_agent = create_react_agent(
    model=model,
    tools=[web_search],
    name="research_expert",
    prompt="You are a world-class researcher with access to a web_search tool."
)

# ---------------------------
# 3) Supervisor (상위 오케스트레이터)
# ---------------------------
supervisor_workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "Use research_expert for information gathering. "
        "Use math_expert for calculations. "
        "When ready, return a concise final answer."
    ),
)
app = supervisor_workflow.compile(checkpointer=InMemorySaver())

# ---------------------------
# 4) 실행 어댑터: Supervisor ↔ 전역 풀(블랙보드) 동기화
# ---------------------------
def supervisor_step(state: GlobalState) -> GlobalState:
    """
    - 전역 풀의 대화 뷰를 Supervisor에 입력
    - Supervisor가 생성한 전체 messages를 받아
      - 마지막 assistant 응답을 블랙보드에 append
      - (선택) 중간 에이전트/툴 메세지도 메타와 함께 append
    """
    # 4-1) 전역 풀 → Supervisor 입력 뷰
    messages_view = pool_view_for_supervisor(state)
    if not messages_view:
        # 최초 실행이면, 시스템 안내를 블랙보드에 심어둔다(선택)
        state = GlobalState(pool=[SystemMessage(content="Global pool init")], profile={}, metrics={})
        messages_view = pool_view_for_supervisor(state)

    # 4-2) Supervisor 실행
    result = app.invoke({"messages": messages_view})
    sup_msgs = result.get("messages", [])

    # 4-3) Supervisor 결과를 블랙보드로 반영
    # - 마지막 assistant 메시지를 우선 append
    last_assistant = next((m for m in reversed(sup_msgs) if m.get("role") == "assistant"), None)
    if last_assistant:
        state.update(pool_append(
            last_assistant["content"],
            stage="answer", route="supervisor", scope="public"
        ))

    # (선택) 중간 체인/툴콜/에이전트 메시지를 메타와 함께 기록하고 싶다면 아래 주석 해제
    # for m in sup_msgs:
    #     if m.get("role") == "tool":
    #         state.update(pool_append(
    #             f"[tool] {m.get('content','')}",
    #             stage="tool", route=m.get("name","?"), scope="private"
    #         ))

    return state

# ---------------------------
# 5) 데모 실행
# ---------------------------
if __name__ == "__main__":
    # 초기 전역 상태(블랙보드)
    state: GlobalState = {
        "pool": [
            HumanMessage(content="what's 12 * 7 plus the length of 'FAANG'?")
        ],
        "profile": {"reading_level": "concise"},
        "metrics": {}
    }

    # 1턴 실행
    state = supervisor_step(state)
    print("=== Global Pool After Turn 1 ===")
    for m in state["pool"]:
        role = type(m).__name__.replace("Message", "").lower()
        meta = (m.additional_kwargs or {}).get("metadata", {})
        print(f"- {role}: {m.content}  | meta={meta}")

    # 2턴: 사용자 추가 질의 → 동일 전역 풀에 누적
    state["pool"].append(HumanMessage(content="and add 3 more. show your final number only."))
    state = supervisor_step(state)
    print("
=== Global Pool After Turn 2 ===")
    for m in state["pool"]:
        role = type(m).__name__.replace("Message", "").lower()
        meta = (m.additional_kwargs or {}).get("metadata", {})
        print(f"- {role}: {m.content}  | meta={meta}")

## 포인트 설명
*   **SSOT 블랙보드**: `pool: Annotated[List[AnyMessage], add_messages]`로 전역 메시지 허브를 한 곳에서 보존합니다. 모든 결과는 `pool_append()`로 append-only로 기록해 감사/재현이 용이합니다.
*   **뷰 분리**: `pool_view_for_supervisor()`에서 Supervisor/에이전트가 쓸 가시성 뷰를 만들어 전달합니다(필요 시 scope/topic 기반 필터 추가).
*   **양방향 동기화**: Supervisor 실행 결과의 최종 assistant 응답(그리고 필요하면 툴/중간 응답)도 다시 블랙보드에 적재합니다. 이렇게 하면 이후 단계(랭킹/피드백/적응)에서 전역 맥락만 보면 됩니다.
*   **확장 지점**:
    *   Swarm 서브그래프를 Retrieval 단계에 붙일 경우, `supervisor_step()` 안에서 Supervisor 대신 Swarm을 호출하고 결과들을 동일한 방식으로 `pool_append()` 하시면 됩니다.
    *   체크포인터는 `InMemorySaver` 대신 `SqliteSaver`/`PostgresSaver`로 교체해 세션 간 지속성을 확보하세요.
    *   운영 규약(Contract): 메시지 `metadata` 표준(예: `stage/route/score/from/to/scope/topic/ts`)을 강제하여 라우팅·품질·보안 규칙을 한 곳에서 구현할 수 있습니다.
