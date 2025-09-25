# PersonaRAG 최종 구현 가이드: `langgraph-supervisor` 공식 패턴 (v5.2)

## 1. 개요

### 1.1. 목표
이 가이드는 **`langgraph-supervisor`** 라이브러리의 공식적인 사용법에 맞춰, LLM 기반의 감독자(Supervisor)가 전체 워크플로우를 지능적으로 제어하는 PersonaRAG 시스템을 구축하는 방법을 안내합니다.

### 1.2. 핵심 아키텍처: 지능형 Supervisor
*   **Supervisor (감독자)**: **LLM으로 구동되는 중앙 관제탑**입니다. 사용자의 요청을 자연어로 이해하고, 각 Worker의 전문성을 바탕으로 다음에 실행할 Worker를 지능적으로 선택합니다.
*   **Workers (작업자)**: 특정 전문 작업을 수행하는 에이전트입니다. 각 Worker는 자신만의 도구(Tools)를 가질 수 있습니다.

## 2. 환경 설정

```bash
pip install langgraph langgraph-supervisor langchain-core langchain-openai
```
*API 키 설정: `os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"`*

## 3. 1단계: 상태 정의
전체 그래프에서 공유될 상태를 정의합니다. `messages`가 모든 에이전트의 대화를 담는 블랙보드 역할을 합니다.

```python
from typing import TypedDict, Annotated, List
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
```

## 4. 2단계: 도구 및 워커(Worker) 에이전트 정의
각 Worker가 사용할 도구와, Worker 자체를 정의합니다. `langgraph.prebuilt`의 `create_react_agent`를 사용하면 간단하게 도구 사용 에이전트를 만들 수 있습니다.

```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# LLM 정의
llm = ChatOpenAI(model="gpt-4o")

# Worker들이 사용할 도구 정의
@tool
def user_profile_tool(user_id: str) -> dict:
    """사용자의 프로필 정보를 조회합니다."""
    print(f"--- Tool: Looking up profile for {user_id} ---")
    # 실제로는 DB 조회
    return {"expertise": "intermediate", "interests": ["AI", "LangGraph"]}

@tool
def retrieval_tool(query: str) -> str:
    """정보를 검색합니다."""
    print(f"--- Tool: Retrieving documents for '{query}' ---")
    return "[Retrieved Docs] 1. Intro to LangGraph, 2. Supervisor Pattern..."

# Worker 에이전트 생성
# 각 Worker는 자신만의 프롬프트와 사용할 도구를 가집니다.
profile_agent = create_react_agent(
    llm,
    tools=[user_profile_tool],
    name="profile_expert",
    prompt="You are a user profile expert. Use the user_profile_tool to fetch user information."
)

retrieval_agent = create_react_agent(
    llm,
    tools=[retrieval_tool],
    name="retrieval_expert",
    prompt="You are a retrieval expert. Use the retrieval_tool to find relevant documents."
)
```

## 5. 3단계: `create_supervisor`로 그래프 조립
`create_supervisor` 함수에 Worker 에이전트 리스트와 **감독자 자신을 위한 프롬프트**를 제공하여 전체 그래프를 생성합니다.

```python
from langgraph_supervisor import create_supervisor

# Supervisor 그래프 생성
# 핵심: Supervisor에게 자연어 프롬프트로 각 Worker의 역할과 사용법을 알려줍니다.
supervisor_graph = create_supervisor(
    llm=llm,
    agents=[profile_agent, retrieval_agent],
    prompt=(
        "You are a supervisor of a team of experts. Your team includes a profile_expert and a retrieval_expert. "
        "Given the user's request, first delegate to the profile_expert to understand the user. "
        "Then, delegate to the retrieval_expert to get information. "
        "When the task is complete and you have the final answer, respond with 'FINISH'."
    )
)

# 컴파일
graph = supervisor_graph.compile()
```

## 6. 4단계: 실행
사용자의 질문을 `HumanMessage`로 담아 그래프를 실행하면, Supervisor LLM이 프롬프트 지시에 따라 지능적으로 Worker들을 호출합니다.

```python
from langchain_core.messages import HumanMessage

# 실행
config = {"configurable": {"thread_id": "persona-rag-v5.2-user-1"}}
initial_input = {"messages": [HumanMessage(content="Based on my profile (user_id: skiiwoo), what is the Supervisor pattern?")]}

print("🚀 Starting PersonaRAG Graph with official Supervisor Pattern...")

# stream으로 각 단계를 확인
for step in graph.stream(initial_input, config, stream_mode="values"):
    print("\n" + "="*40)
    print(f"Step Output: {step}")
    print("="*40)

final_response = graph.get_state(config)
print("\n🏁 Final Answer:", final_response.values['messages'][-1].content)
```

## 7. 결론
`langgraph-supervisor` 라이브러리의 공식 패턴은 **LLM의 추론 능력 자체를 오케스트레이션에 활용**하는 매우 강력하고 직관적인 방법을 제공합니다. 개발자는 복잡한 라우팅 규칙을 코드로 작성하는 대신, **감독자에게 자연어로 된 지시사항(프롬프트)**을 내리기만 하면 됩니다.

이 방식은 우리가 이전에 논의했던 수동 라우팅 방식보다 훨씬 더 유연하며, LangChain 팀이 지향하는 'LLM-as-Orchestrator' 철학을 가장 잘 반영한 최종 구현안입니다.
