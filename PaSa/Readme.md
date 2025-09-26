# PaSa 구현을 위한 `langgraph-supervisor` 최종 가이드 (v3)

## 1. 개요 및 설치

이 문서는 `langgraph-supervisor`의 `create_supervisor` API를 사용하여, **미리 빌드된 ReAct 에이전트**와 **커스텀 StateGraph 서브그래프**를 함께 조율하는 PaSa 구현 최종 가이드를 제공합니다.

### 설치

```bash
pip install langgraph-supervisor langgraph langchain-openai
```

## 2. PaSa 구현 전체 코드

### 2.1. 상태(State) 스키마 정의

```python
from typing import TypedDict, List, Optional, Dict, Any, Set

class Paper(TypedDict):
    id: str
    title: str
    abstract: str
    select_score: float
    depth: int

class PaSaState(TypedDict):
    query: str
    paper_queue: List[Paper]
    current_paper: Optional[Paper]
    processed_ids: Set[str]
    budget: Dict[str, int]
    last_error: Optional[str]
    messages: Any
```

### 2.2. 도구(Tools) 정의

`crawler`와 `selector`가 사용할 도구들을 정의합니다.

```python
from langchain_core.tools import tool

@tool
def search_tool(query: str) -> List[Paper]:
    """논문 검색 도구"""
    # ...
    return []

@tool
def expand_tool(paper_id: str) -> List[Paper]:
    """논문 확장(인용) 도구"""
    # ...
    return []

@tool
def select_tool(paper_title: str, paper_abstract: str) -> dict:
    """논문 평가 도구"""
    # ...
    return {}
```

### 2.3. Worker 에이전트 정의

#### 2.3.1. ReAct 에이전트: `crawler` & `selector`

도구를 사용하여 외부와 상호작용하는 에이전트들을 `create_react_agent`로 생성합니다.

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o", temperature=0)

crawler = create_react_agent(model, tools=[search_tool, expand_tool], name="crawler")
selector = create_react_agent(model, tools=[select_tool], name="selector")
```

#### 2.3.2. 커스텀 서브그래프 에이전트: `manager`

큐 정렬, 예산 확인 등 복잡한 내부 상태 관리 로직은 별도의 `StateGraph`로 만듭니다. 이것이 바로 **커스텀 에이전트**가 됩니다.

```python
from langgraph.graph import StateGraph

def prioritize_queue_node(state: PaSaState) -> dict:
    """우선순위 큐를 정렬하고, 다음 작업 논문을 선택합니다."""
    sorted_queue = sorted(state["paper_queue"], key=lambda p: p.get("select_score", 0.0), reverse=True)
    next_paper = sorted_queue.pop(0) if sorted_queue else None
    return {"paper_queue": sorted_queue, "current_paper": next_paper}

def check_budget_node(state: PaSaState) -> dict:
    """예산을 확인하고, 소진 시 current_paper를 None으로 설정하여 중단 신호를 보냅니다."""
    if state["budget"]["expand"] <= 0:
        return {"current_paper": None} # 예산 소진
    return {}

# 상태 관리 로직을 담은 StateGraph 생성
management_graph = StateGraph(PaSaState)
management_graph.add_node("prioritize", prioritize_queue_node)
management_graph.add_node("check_budget", check_budget_node)

# 노드 연결
management_graph.set_entry_point("prioritize")
management_graph.add_edge("prioritize", "check_budget")
management_graph.set_finish_point("check_budget")

# Supervisor에 등록할 수 있도록 이름(name)을 부여하여 컴파일
manager_agent = management_graph.compile(name="manager")
```

### 2.4. Supervisor 정의 및 실행

이제 모든 에이전트(`crawler`, `selector`, `manager`)를 `create_supervisor`에 등록합니다.

```python
from langgraph_supervisor import create_supervisor

# Supervisor가 에이전트를 선택하기 위한 라우팅 프롬프트
routing_prompt = """
당신은 연구 프로젝트를 총괄하는 Supervisor입니다. 현재 상태를 분석하여 다음에 어떤 에이전트를 호출할지 결정하세요.

상태 요약:
- 남은 예산: {budget}
- 대기열 논문 수: {paper_queue_size}
- 현재 작업 논문: {current_paper_title}

결정 규칙:
1.  **초기 검색**: 대기열이 비어있고 검색 예산이 남았다면, `crawler`를 호출하여 검색하세요.
2.  **평가**: `selector`가 아직 평가하지 않은 새 논문이 있다면, `selector`를 호출하여 평가하세요.
3.  **상태 관리**: 논문 평가가 끝났거나, 새 논문이 추가되었다면, `manager`를 호출하여 큐를 정렬하고 다음 작업을 준비하세요.
4.  **확장**: `manager`가 다음 작업 논문(`current_paper`)을 성공적으로 선택했고 확장 예산이 남았다면, `crawler`를 호출하여 확장하세요.
5.  **종료**: 더 이상 진행할 작업(대기열, 예산)이 없다면, `FINISH`를 반환하여 작업을 종료하세요.

정확히 다음 중 하나만 반환해야 합니다: crawler / selector / manager / FINISH
"""

# Supervisor 생성
supervisor = create_supervisor(
    # 프리빌트 ReAct 에이전트와 커스텀 서브그래프 에이전트를 함께 등록
    agents=[crawler, selector, manager_agent],
    model=model,
    prompt=routing_prompt,
    state_schema=PaSaState,
)

workflow = supervisor.compile()

# --- 실행 예시 ---
# init_state = { ... } (이전과 동일)
# final_state = workflow.invoke({"messages": [("user", init_state["query"])], **init_state})
```

## 3. 최종 설계의 장점

- **역할 분리**: `crawler`와 `selector`는 외부와 상호작용하는 역할에, `manager`는 내부 상태를 관리하는 역할에 집중하여 각 컴포넌트가 단순하고 명확해집니다.
- **유연한 제어**: Supervisor는 도구(Tool)가 아닌, 더 큰 작업 단위인 에이전트(Agent)를 호출하므로 라우팅 프롬프트가 더 간결하고 직관적이 됩니다.
- **확장성**: '체크포인팅 에이전트', '요약 보고서 생성 에이전트' 등 새로운 커스텀 서브그래프를 만들어 `agents` 리스트에 추가하기만 하면 손쉽게 기능을 확장할 수 있습니다.
