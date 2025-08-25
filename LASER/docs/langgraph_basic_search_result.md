# LangGraph 에이전트: 기본 그래프(Search → Result) 구현 가이드

이 문서는 LangGraph로 가장 단순한 형태의 "검색 → 결과" 에이전트 그래프를 구현하는 방법을 단계별로 설명합니다. 최소 동작 예제를 통해 시작한 뒤, 점진적으로 확장하는 포인트(스트리밍, 체크포인팅, 재시도 등)를 제시합니다.

## 1) 개요
- 목표: 사용자의 `query`를 입력받아, 검색 도구를 호출하고, 간단히 결과를 요약/정리해 반환하는 기본 그래프 구성
- 노드 구성
  - `search`: 외부 검색 도구(예: DuckDuckGo) 호출, 상위 N개 결과를 상태에 저장
  - `result`: 검색 결과를 사용자 보기 좋게 포매팅해 최종 응답으로 저장
- 그래프 흐름: `START` → `search` → `result` → `END`

## 2) 준비사항
- Python 3.9+
- 패키지 설치
  ```bash
  pip install "langgraph>=0.2.0" "langchain-core>=0.2.0" duckduckgo-search typing_extensions
  ```
  - 이 가이드는 API 키가 필요 없는 `duckduckgo-search`를 예제로 사용합니다.

## 3) 상태(state) 설계
LangGraph에서는 상태를 명시적으로 정의해 노드 간 데이터 전달을 타입 안정적으로 관리할 수 있습니다.

```python
from typing import List, Optional
from typing_extensions import TypedDict, NotRequired

class SearchState(TypedDict):
    # 입력
    query: str

    # 중간/최종 산출물
    results: NotRequired[List[dict]]  # 검색 결과 리스트
    answer: NotRequired[str]          # 최종 응답 텍스트
    error: NotRequired[str]           # 오류 메시지(선택)
```

- `NotRequired`를 사용해 초기 입력 시 반드시 필요하지 않은 키를 정의합니다.

## 4) 노드 구현
두 개의 순수 함수형 노드를 정의합니다. 각 노드는 `state`를 입력받아 부분 상태를 반환(머지)합니다.

```python
from duckduckgo_search import DDGS

# 검색 노드: query로 DuckDuckGo 검색 후 상위 N개 결과를 저장
def search_node(state: SearchState) -> dict:
    query = state["query"].strip()
    if not query:
        return {"error": "Empty query provided.", "results": []}

    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, region="wt-wt", safesearch="moderate", max_results=5))
        # hits: [{"title": str, "href": str, "body": str, ...}, ...]
        results = [
            {
                "title": h.get("title"),
                "url": h.get("href"),
                "snippet": h.get("body"),
            }
            for h in hits
        ]
        return {"results": results}
    except Exception as e:
        return {"error": f"search_failed: {e}", "results": []}

# 결과 노드: results를 간단히 포매팅하여 최종 답변 생성
def result_node(state: SearchState) -> dict:
    results = state.get("results", []) or []
    error = state.get("error")

    # 간단한 포매팅
    lines = []
    if error:
        lines.append(f"[주의] {error}")
    if not results:
        lines.append("검색 결과가 비어 있습니다.")
    else:
        lines.append("검색 결과 상위 목록:")
        for i, r in enumerate(results, 1):
            title = r.get("title") or "(제목 없음)"
            url = r.get("url") or ""
            snippet = (r.get("snippet") or "").strip()
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            lines.append(f"{i}. {title}\n   {url}\n   {snippet}")

    answer = "\n".join(lines)
    return {"answer": answer}
```

## 5) 그래프 조립(StateGraph)
`START` → `search` → `result` → `END` 순서로 단순 연결합니다.

```python
from langgraph.graph import StateGraph, START, END

def build_graph():
    graph = StateGraph(SearchState)
    graph.add_node("search", search_node)
    graph.add_node("result", result_node)

    graph.add_edge(START, "search")
    graph.add_edge("search", "result")
    graph.add_edge("result", END)

    return graph.compile()
```

## 6) 실행 예시
하나의 파일(`basic_search_result.py`)로 합쳐 실행하는 예시입니다.

```python
# basic_search_result.py
from typing import List
from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END
from duckduckgo_search import DDGS

class SearchState(TypedDict):
    query: str
    results: NotRequired[List[dict]]
    answer: NotRequired[str]
    error: NotRequired[str]

# --- Nodes ---

def search_node(state: SearchState) -> dict:
    query = state["query"].strip()
    if not query:
        return {"error": "Empty query provided.", "results": []}
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, region="wt-wt", safesearch="moderate", max_results=5))
        results = [
            {
                "title": h.get("title"),
                "url": h.get("href"),
                "snippet": h.get("body"),
            }
            for h in hits
        ]
        return {"results": results}
    except Exception as e:
        return {"error": f"search_failed: {e}", "results": []}


def result_node(state: SearchState) -> dict:
    results = state.get("results", []) or []
    error = state.get("error")

    lines = []
    if error:
        lines.append(f"[주의] {error}")
    if not results:
        lines.append("검색 결과가 비어 있습니다.")
    else:
        lines.append("검색 결과 상위 목록:")
        for i, r in enumerate(results, 1):
            title = r.get("title") or "(제목 없음)"
            url = r.get("url") or ""
            snippet = (r.get("snippet") or "").strip()
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            lines.append(f"{i}. {title}\n   {url}\n   {snippet}")

    return {"answer": "\n".join(lines)}

# --- Build & Run ---

def build_graph():
    g = StateGraph(SearchState)
    g.add_node("search", search_node)
    g.add_node("result", result_node)
    g.add_edge(START, "search")
    g.add_edge("search", "result")
    g.add_edge("result", END)
    return g.compile()

if __name__ == "__main__":
    graph = build_graph()

    # 단일 호출
    final_state = graph.invoke({"query": "LangGraph 튜토리얼"})
    print("\n[최종 응답]\n" + final_state.get("answer", "(비어 있음)"))

    # 스트리밍으로 내부 노드 이벤트 확인
    print("\n[스트리밍 이벤트]")
    for event in graph.stream({"query": "파이썬 뉴스"}):
        # event는 {node_name: state_delta} 구조의 딕셔너리 시퀀스
        print(event)
```

실행:
```bash
python basic_search_result.py
```

## 7) 스트리밍으로 디버깅/관찰
- `graph.stream(input)`은 각 노드의 상태 변경을 순서대로 방출합니다.
- 간단히 `print(event)`만으로도 어느 노드가 어떤 키를 업데이트했는지 빠르게 파악할 수 있습니다.

## 8) 확장 포인트
- Tool 다양화: 검색 외에 웹 스크래핑, 벡터 검색, 사내 검색 등으로 교체/추가
- LLM 후처리: `result` 노드에서 LLM을 사용해 요약/정제(예: OpenAI, local LLM 등). LangChain Runnable과 결합 가능
- 조건 분기(Conditional edges): 결과가 비었을 때 재검색, 쿼리 보정 노드로 분기
- 재시도/회로 차단: 검색 실패 시 백오프 재시도, 일정 횟수 이상 실패 시 종료
- 체크포인팅: `langgraph.checkpoint.memory.MemorySaver` 등으로 상태 저장/복원
- 병렬화: 여러 검색 엔진 노드를 병렬 실행 후 머지
- 평가/테스트: 고정 쿼리 세트에 대한 리그레션 테스트와 스냅샷 비교

## 9) 트러블슈팅
- 네트워크 차단 환경: `duckduckgo-search`가 외부 연결을 필요로 하므로, 오프라인 환경에서는 로컬 인덱스나 모의(mock) 검색 함수를 사용하세요.
- 빈 쿼리: 본 예제는 빈 문자열을 오류로 처리합니다. UX 요구에 따라 기본 키워드 제안 노드를 두는 것도 방법입니다.
- 인코딩/문자셋: 결과 문자열 포매팅 시 이모지/특수문자 포함 여부에 주의하세요.

## 10) 요약
- 상태를 명시하고, 노드를 순수 함수로 만들고, `StateGraph`로 연결하면 작은 그래프를 빠르게 구축할 수 있습니다.
- 본 가이드는 최소 동작 예시입니다. 조건 분기, 재시도, 체크포인팅 등을 추가하며 점진적으로 확장해 보세요.
