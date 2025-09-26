### **PersonaRAG 구현 가이드 (세분화된 에이전트 패턴 v5.3)**

**목표:** v3.0 아키텍처의 각 핵심 기능을 독립된 워커 에이전트로 구현하고, Supervisor가 이들을 정교하게 지휘하는 최종 시스템을 구축합니다.

---

#### **1. 아키텍처 변경점: 세분화된 전문가 모델**

*   **이전 (v5.2):** `profile_manager`, `information_retriever` 등 여러 도구를 가진 **범용 전문가** 모델
*   **변경 (v5.3):** `user_profile_agent`, `retrieval_agent`, `ranking_agent`, `feedback_agent` 등 **단일 책임**을 가진 **세분화된 전문가** 모델

`Live Session`은 `AgentState`의 `messages` 리스트로서, 모든 에이전트가 자신의 작업 기록과 결과를 게시하고 다른 에이전트의 작업을 참조하는 '공용 블랙보드' 역할을 합니다.

---

#### **2. 1단계: 도구 정의 (`persona_rag/tools.py`)**

도구 정의는 이전과 동일합니다. 각 도구는 하나의 명확한 기능을 수행합니다.

```python
# persona_rag/tools.py
# (v5.2 가이드와 동일한 내용)

import json
from langchain_core.tools import tool

@tool
def get_user_profile(user_id: str) -> dict:
    """사용자 ID를 기반으로 영속적인 사용자 프로필을 조회합니다."""
    # ... (구현은 이전과 동일)
    print(f"--- Tool: get_user_profile(user_id='{user_id}') ---")
    try:
        with open(f"profiles/user_{user_id}.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "User profile not found."}


@tool
def update_user_profile(user_id: str, feedback_summary: str) -> str:
    """사용자 피드백 요약을 바탕으로 사용자 프로필을 업데이트합니다."""
    # ... (구현은 이전과 동일)
    print(f"--- Tool: update_user_profile(user_id='{user_id}', feedback='{feedback_summary}') ---")
    profile = get_user_profile(user_id)
    profile.setdefault("notes", []).append(feedback_summary)
    with open(f"profiles/user_{user_id}.json", "w") as f:
        json.dump(profile, f, indent=2)
    return f"Profile for {user_id} updated successfully."

@tool
def contextual_retrieval(query: str, user_profile: dict) -> list[str]:
    """사용자 프로필과 쿼리를 바탕으로 문맥에 맞는 정보를 검색합니다."""
    # ... (구현은 이전과 동일)
    print(f"--- Tool: contextual_retrieval(query='{query}') ---")
    return [
        "LangGraph는 복잡한 LLM 워크플로우를 만들기 위한 라이브러리입니다.",
        "Supervisor 패턴은 LLM을 오케스트레이터로 사용합니다.",
        "Cross-encoder 모델은 문서 순위 재조정에 효과적입니다."
    ]

@tool
def document_ranking(documents: list[str], query: str) -> list[str]:
    """검색된 문서 목록을 사용자의 질문과의 관련성을 기준으로 순위를 재조정합니다."""
    # ... (구현은 이전과 동일)
    print(f"--- Tool: document_ranking(query='{query}') ---")
    documents.reverse()
    return documents
```

---

#### **3. 2단계: 세분화된 워커 에이전트 생성 (`persona_rag/agents.py`)**

**이 단계가 핵심적인 변경점입니다.** 각 에이전트는 이제 단 하나의 도구와 단 하나의 책임을 가집니다.

```python
# persona_rag/agents.py

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from persona_rag import tools

# 공통 LLM 정의
llm = ChatOpenAI(model="gpt-4o")

# --- 각 기능을 독립된 에이전트로 정의 ---

user_profile_agent = create_react_agent(
    llm,
    tools=[tools.get_user_profile],
    name="user_profile_agent",
    prompt="You are a user profile specialist. Your sole job is to fetch a user's profile using the provided tool."
)

retrieval_agent = create_react_agent(
    llm,
    tools=[tools.contextual_retrieval],
    name="retrieval_agent",
    prompt="You are a retrieval specialist. Your sole job is to find documents based on a query and user profile."
)

ranking_agent = create_react_agent(
    llm,
    tools=[tools.document_ranking],
    name="ranking_agent",
    prompt="You are a ranking specialist. Your sole job is to re-rank a list of documents based on relevance to a query."
)

feedback_agent = create_react_agent(
    llm,
    tools=[tools.update_user_profile],
    name="feedback_agent",
    prompt="You are a feedback specialist. Your sole job is to update a user's profile based on a summary of their feedback."
)
```

---

#### **4. 3단계: Supervisor 그래프 조립 (`persona_rag/graph.py`)**

Supervisor의 프롬프트가 더 길고 정교해집니다. 이제 여러 명의 단일 기능 전문가들을 순서대로 지휘해야 합니다.

```python
# persona_rag/graph.py

from langgraph_supervisor import create_supervisor
from persona_rag.agents import (
    llm,
    user_profile_agent,
    retrieval_agent,
    ranking_agent,
    feedback_agent
)
from persona_rag.state import AgentState

# Supervisor에게 더 많은 워커들을 소개하고, 더 상세한 작업 흐름을 지시합니다.
supervisor_graph = create_supervisor(
    llm=llm,
    agents=[
        user_profile_agent,
        retrieval_agent,
        ranking_agent,
        feedback_agent
    ],
    state_schema=AgentState,
    prompt=(
        "You are a meticulous project manager supervising a team of highly specialized AI agents.\n"
        "Your team consists of:\n"
        "- user_profile_agent: Only fetches user profile data.\n"
        "- retrieval_agent: Only retrieves documents.\n"
