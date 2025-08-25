# LASER LangGraph: 상태별 시스템 지침과 함수 정의 예시

이 문서는 LASER 에이전트를 LangGraph로 구현할 때, LLM(function-calling)을 통한 의사결정을 위해 사용할 수 있는 상태별 시스템 지침(system instruction)과 함수 정의(tools) 초안을 제공합니다. 프롬프트와 함수 정의는 작업/도메인에 맞게 조정하세요.

## 공통 원칙
- 모든 상태에서 Thought(생각)를 먼저 기술하고, 그 다음 허용된 함수 중 하나를 호출합니다.
- 함수 인수는 가능한 한 최소/정규화된 정보만 포함합니다.
- 상태별로 허용되지 않은 액션은 호출하지 않도록 시스템 지침에서 명시합니다.

## Search 상태
- 목적: 사용자 지시를 바탕으로 검색 키워드를 생성하고 검색을 수행합니다.

시스템 지침 템플릿
```
당신은 WebShop 웹 환경에서 쇼핑을 돕는 LASER 에이전트입니다.
지금은 Search 상태입니다. 이 상태에서는 검색 키워드를 생성하고 검색을 수행하세요.
규칙:
- 먼저 Thought에 근거를 1~2문장으로 적습니다.
- 이어서 search 키워드를 함수 호출로 실행합니다.
- 허용 함수 외에는 호출하지 마십시오.
출력 형식:
- 반드시 함수 호출을 1회 포함해야 합니다.
```

허용 함수 정의
```json
[
  {
    "name": "do_search",
    "description": "키워드로 상품 검색을 수행",
    "parameters": {
      "type": "object",
      "properties": {
        "keywords": {
          "type": "array",
          "items": {"type": "string"},
          "description": "검색 키워드 토큰 리스트"
        }
      },
      "required": ["keywords"]
    }
  }
]
```

후처리 매핑 예시
- 모델이 `do_search({"keywords": ["wireless", "mouse"]})`를 호출하면, env 액션 문자열은 `search[wireless mouse]`로 변환해 `env.step`에 전달합니다.

## Result 상태
- 목적: 검색 결과 목록에서 아이템을 선택하거나 다음/이전 페이지로 넘기거나 검색으로 돌아갑니다. 필요 시 유망한 결과를 메모리 버퍼에 저장합니다.

시스템 지침 템플릿
```
지금은 Result 상태입니다. 다음 중 하나를 수행하세요:
- select_item: 가장 적합한 상품을 선택(상품 라벨/ASIN 텍스트를 인수로 지정)
- go_next: 다음 검색 결과 페이지로 이동
- back_to_search: 검색 페이지로 돌아가기
규칙:
- 먼저 Thought로 선택 근거를 1~2문장 기술
- 함수 호출은 정확히 1회
- 상품이 불충분하게 맞지만 유망하면 memory_buffer에 기록한 뒤 go_next를 선택할 수 있습니다.
```

허용 함수 정의
```json
[
  {
    "name": "select_item",
    "description": "검색 결과에서 특정 상품(라벨/ASIN)을 선택",
    "parameters": {
      "type": "object",
      "properties": {
        "label": {"type": "string", "description": "상품을 식별하는 클릭 라벨 텍스트"}
      },
      "required": ["label"]
    }
  },
  {"name": "go_next", "description": "다음 결과 페이지로 이동", "parameters": {"type": "object", "properties": {}}},
  {"name": "back_to_search", "description": "검색 페이지로 돌아가기", "parameters": {"type": "object", "properties": {}}}
]
```

후처리 매핑 예시
- `select_item({"label": "B01ABCD"})` → `click[B01ABCD]`
- `go_next({})` → `click[next page]`
- `back_to_search({})` → `click[back to search]` 또는 환경에 맞는 액션

## Item 상태
- 목적: 상세/서브 페이지 확인(Description/Features/Reviews), 이전 페이지로 이동, 구매 결정(Buy Now).

시스템 지침 템플릿
```
지금은 Item 상태입니다. 다음 중 하나를 수행하세요:
- open_subpage: description/features/reviews 중 하나 열기
- buy_now: 구매를 확정하고 종료 상태로 이동
- go_prev: 이전 페이지로 돌아가기
규칙:
- Thought를 1~2문장으로 작성
- 허용 함수 1회 호출
```

허용 함수 정의
```json
[
  {
    "name": "open_subpage",
    "description": "아이템 서브 페이지 열기",
    "parameters": {
      "type": "object",
      "properties": {
        "name": {
          "type": "string",
          "enum": ["description", "features", "reviews"],
          "description": "열고 싶은 서브페이지"
        }
      },
      "required": ["name"]
    }
  },
  {"name": "buy_now", "description": "바로 구매하고 종료 상태로 이동", "parameters": {"type": "object", "properties": {}}},
  {"name": "go_prev", "description": "이전 페이지로 돌아가기", "parameters": {"type": "object", "properties": {}}}
]
```

후처리 매핑 예시
- `open_subpage({"name": "description"})` → `click[description]`
- `buy_now({})` → `click[buy now]`
- `go_prev({})` → `click[< prev]` (환경에서 정의된 정확한 라벨 사용)

## 샘플 LangChain 연동 코드 스니펫
```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

model = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=False)

def call_llm(state, system_prompt: str, tools: list, human_input: str):
    # langchain의 tool_call 포맷 사용을 가정
    from langchain_core.pydantic_v1 import BaseModel
    # tools는 model.bind_tools(tools=tool_schemas)로 전달
    bound = model.bind_tools(tools)
    resp = bound.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_input),
    ])
    # resp.tool_calls[0]에 name, args 존재
    if not resp.tool_calls:
        raise ValueError("No tool call returned")
    tc = resp.tool_calls[0]
    return tc["name"], tc["args"]
```

주의
- 환경의 실제 클릭 라벨은 `web_agent_site/envs/web_agent_text_env.py`의 파싱 결과에 따라 달라집니다. `[button] ... [button_]` 사이 텍스트를 정확히 사용해야 합니다.
- 모델 프롬프트에 현재 관찰(obs)와 허용 액션 요약, 제약을 포함해 일관된 행동을 유도하세요.
