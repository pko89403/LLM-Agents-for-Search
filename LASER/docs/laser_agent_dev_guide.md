## LASER 에이전트 `langgraph` 기반 개발 설계 문서

### 1. 개요
본 문서는 웹 탐색 작업을 위한 LLM 에이전트인 LASER(LLM Agent with State-Space ExploRation)를 `langgraph` 프레임워크를 활용하여 개발하기 위한 설계 방안을 제시합니다. LASER는 기존 LLM 에이전트의 "전방향 실행 모드(forward-only execution mode)" 및 "글로벌 액션 공간(global action space)"의 한계를 극복하기 위해, 상호작용 작업을 **상태 공간 탐색(state-space exploration)**으로 모델링하며 유연한 오류 복구 및 유효한 액션 선택 능력을 제공합니다. `langgraph`는 그래프 기반의 상태 관리 및 전환을 통해 복잡한 에이전트 워크플로우를 효과적으로 구현할 수 있는 도구이므로, LASER의 핵심 설계 원칙과 잘 부합합니다.

### 2. LASER 아키텍처 핵심 원칙
LASER의 핵심 아키텍처는 다음과 같은 원칙들을 기반으로 합니다:
*   **상태 공간 탐색 모델링**: 웹 탐색 작업을 에이전트가 미리 정의된 상태들 사이를 전환하며 작업을 완료하는 **상태 공간 탐색**으로 모델링합니다. 이를 통해 잘못된 액션으로부터 쉽게 복구할 수 있는 **유연한 백트래킹(flexible backtracking)**이 가능해집니다.
*   **상태 기반 액션 공간(State-Specific Action Space)**: 각 상태에 따라 허용되는 액션 공간을 명확하게 정의하여, 에이전트가 항상 **유효한 액션**을 선택하도록 보장하고 작업의 난이도를 줄입니다. LASER는 실험에서 **100% 유효한 액션**을 수행했습니다.
*   **제로-샷 학습 및 상태별 시스템 지침**: 인컨텍스트 예시(in-context examples)를 사용하지 않고, 각 상태에서 에이전트가 마주칠 수 있는 상황과 이를 처리하는 방법을 알려주는 **상세한 상태별 시스템 지침(state-specific system instruction)**만을 제공하여 에이전트의 탐색을 안내합니다.
*   **생각-액션 프로세스(Thought-and-Action Process)**: 매 단계마다 "생각(thought)"을 생성한 후 그 생각에 기반하여 "행동(action)"을 선택하는 과정을 반복합니다.
*   **메모리 버퍼(Memory Buffer)**: 탐색 과정에서 검토했지만 사용자 지침과 완벽히 일치하지 않는다고 판단한 **중간 결과들을 저장**합니다. 이는 에이전트가 최대 탐색 단계에 도달했을 때 유연성을 제공하는 **"백업 전략(backup strategy)"의 기반**이 됩니다.

### 3. `langgraph` 기반 설계
LASER의 `langgraph` 구현은 에이전트의 상태, 노드(각 상태), 그리고 이들 사이의 전환(액션)을 명확하게 정의하는 데 중점을 둡니다.

#### 3.1. 그래프 상태 정의 (Graph State Definition)
`langgraph`의 `State`는 에이전트의 현재 정보를 나타내며, 각 노드에서 읽고 쓸 수 있습니다. LASER 에이전트의 `State`는 다음 요소들을 포함할 수 있습니다:
*   **`current_laser_state`**: 현재 LASER 에이전트가 위치한 상태 (예: "Search", "Result", "Item", "Stopping").
*   **`user_instruction`**: 사용자가 에이전트에게 내린 초기 지시.
*   **`current_observation`**: 웹 환경으로부터 받은 현재 관찰 정보 (웹페이지 내용, 버튼 목록 등).
*   **`thought_history`**: 과거의 "생각(thought)" 기록.
*   **`action_history`**: 과거의 "행동(action)" 기록.
*   **`memory_buffer`**: 검토했지만 사용자 지침과 일치하지 않는다고 판단한 중간 결과(아이템) 목록.
*   **`step_count`**: 현재까지 수행된 상태 전환(state transitions) 횟수.
*   **`selected_item`**: 최종적으로 선택된 아이템 (stopping state에서 출력될 정보).
*   **`item_config`**: 최근 검색에 사용한 구성/키워드 등(아이템 판단 보조용).

#### 3.2. 노드(Nodes) 정의 및 역할
각 LASER 상태는 `langgraph`의 개별 노드(Node)로 매핑되며, 각 노드는 특정 상태에서의 로직을 처리합니다. 모든 노드는 "생각-액션 프로세스"를 내부에 포함하며, LLM과 함수 호출 기능을 활용하여 다음 액션을 결정합니다.

*   **`SearchNode` (검색 상태 노드)**:
    *   **역할**: 사용자의 지시에 따라 **검색 쿼리를 생성**하고 'Search' 액션을 수행하여 관련된 항목을 찾습니다.
    *   **입력**: `current_observation`, `user_instruction`, `thought_history`, `action_history`.
    *   **출력**: LLM이 생성한 `thought` 및 `Search` 액션, 업데이트된 `thought_history`, `action_history`.
    *   **다음 상태**: `ResultNode` (검색 결과 페이지로 이동).

*   **`ResultNode` (결과 상태 노드)**:
    *   **역할**: 검색 결과 목록이 표시되는 상태입니다. 사용자 지시에 맞는 항목을 선택하거나, 다음 페이지로 이동하거나, 검색 페이지로 돌아가는 액션을 수행합니다.
    *   **입력**: `current_observation`, `user_instruction`, `thought_history`, `action_history`, `memory_buffer`.
    *   **출력**: LLM이 생성한 `thought` 및 `select_item`, `Next`, `Back_to_Search` 액션 중 하나, 업데이트된 기록 및 `memory_buffer` (불일치 항목 저장).
    *   **다음 상태**: `ItemNode` (항목 선택 시), `ResultNode` (다음 페이지 이동 시), `SearchNode` (검색 페이지로 돌아갈 때), `StoppingNode` (최종 아이템 선정 및 구매 결정 시).

*   **`ItemNode` (항목 상태 노드)**:
    *   **역할**: 특정 항목의 상세 정보를 확인하는 상태입니다. 항목이 사용자 지시에 맞는지 확인하고, 사용자 정의 옵션을 고려하며, 필요한 경우 설명, 기능, 리뷰를 확인하거나 이전 페이지로 돌아가거나 항목을 **구매(Buy_Now)**합니다.
    *   **입력**: `current_observation`, `user_instruction`, `thought_history`, `action_history`, `memory_buffer`.
    *   **출력**: LLM이 생성한 `thought` 및 `Description`, `Features`, `Reviews`, `Buy_Now`, `Prev` 액션 중 하나, 업데이트된 기록 및 `memory_buffer`.
    *   **다음 상태**: `ItemNode` (설명, 기능, 리뷰 확인 시), `ResultNode` (이전 페이지로 돌아갈 때), `StoppingNode` (구매 결정 시).

*   **`StoppingNode` (정지 상태 노드)**:
    *   **역할**: 에이전트가 탐색을 멈추고 최종 결과를 출력하는 상태입니다. 이 상태에서는 더 이상 액션을 수행하지 않고 작업을 완료합니다.
    *   **입력**: `selected_item`, `memory_buffer` (최대 탐색 단계 도달 시 백업 전략으로 사용).
    *   **출력**: 최종 `selected_item`.

*   **`CommonLLMLogicNode` (공통 LLM 로직 노드)**: 각 상태 노드 내부에서 호출될 수 있는 공통 로직.
    *   **역할**: 현재 상태의 시스템 지침과 관찰을 바탕으로 **GPT-4 모델**을 사용하여 "생각"을 생성하고, 정의된 **함수 호출(function-calling) 기능**을 통해 유효한 "액션"을 선택합니다.
    *   **입력**: `current_observation`, `system_instruction` (상태별 지침), `allowed_actions` (상태별 액션 공간), `thought_history`, `action_history`.
    *   **출력**: `next_thought`, `selected_action`.

#### 3.3. 엣지(Edges) 및 전환 조건 정의
`langgraph`는 조건부 엣지(conditional edges)를 통해 에이전트의 흐름을 제어합니다. laser_agent.py의 실제 동작을 기준으로, 관찰 텍스트에 포함된 버튼/페이지 컨텍스트로 상태를 판별합니다.

- Search 상태: 초기 reset 직후 혹은 검색 페이지. 허용 액션은 search_items.
- Result 상태: 관찰에 "[button] Back to Search [button_]"가 보일 때(검색 결과 페이지). 허용 액션은 click_item/select_item, next_page, back_to_search.
- Item 상태: 특정 아이템 클릭 후 상세 페이지. 허용 액션은 Description/Reviews/Features, Buy_Now, Prev.
- Stopping 상태: Buy Now 실행 또는 최대 스텝 도달 시 종료.

전이 규칙
- START → Search: 초기 진입
- Search → Result: 검색 수행 후 고정 전이
- Result:
 - select_item → Item
 - Next → Result(자기 루프)
 - Back_to_Search → Search
- Item:
 - Description/Features/Reviews → Item(자기 루프)
 - Prev → Result
 - Buy_Now → Stopping
- 모든 상태 공통:
 - step_count ≥ MAX_STEPS → Stopping(백업 전략; memory_buffer에서 후보 선택)

#### 3.4. 핵심 컴포넌트 구현 상세

*   **LLM 연동 및 함수 호출**:
    *   LASER는 **GPT-4-0613 모델**을 백본으로 사용하며, **함수 호출(function-calling) 기능**을 활용하여 액션 선택 단계를 구현합니다.
    *   `langgraph` 노드 내에서 LLM 호출 시, 각 상태에서 허용되는 액션들은 **함수 정의 리스트 형태**로 LLM에 전달됩니다. 각 함수는 목적과 인수를 설명하는 짧은 설명을 포함합니다.
    *   LLM의 응답으로 함수 호출이 반환되면, 해당 함수를 실행하여 웹 환경과 상호작용합니다.

*   **메모리 버퍼 관리**:
    *   `memory_buffer`는 `langgraph`의 `State` 내에 리스트 형태로 관리됩니다.
    *   `ResultNode`나 `ItemNode`에서 현재 검토 중인 아이템이 사용자 지침과 **완벽히 일치하지는 않지만 잠재적으로 유효한 경우**, 해당 아이템을 `memory_buffer`에 추가합니다.
    *   최대 탐색 단계 도달 시 `StoppingNode`에서 `memory_buffer`를 검토하여 최적의 백업 아이템을 선택합니다.

*   **상태별 시스템 지침 프롬프트 관리**:
    *   각 노드(Search, Result, Item)는 해당하는 **상세한 시스템 지침 프롬프트**를 로드하고 LLM 호출 시 사용합니다.
    *   이 지침은 에이전트가 예상치 못한 상황에서도 스스로 판단하여 적절한 액션을 취할 수 있도록 돕습니다.

*   **환경 상호작용**:
    *   LASER는 **WebShop 시뮬레이션 환경**에서 평가되었으며, 실제 `langgraph` 노드 내에서는 WebShop 환경과의 API 호출 또는 웹 파싱 로직이 구현되어 `current_observation`을 업데이트합니다.
    *   `web_agent_site` 폴더를 LASER 저장소의 것으로 교체하여 지침의 결정론성을 보장해야 합니다.

#### 3.5. 메모리 버퍼 스키마와 기록 포맷

LASER의 메모리 버퍼는 baseline 코드의 browsed_items(dict)을 일반화한 구성입니다. LangGraph에서는 상태 내 `memory_buffer: List[dict]`로 관리하며, Result/Item 상태에서 유망하지만 아직 최종 선택하지 않은 후보를 축적합니다. Stopping(강제 종료 포함) 시 버퍼를 사용해 백업 전략을 수행합니다.

권장 스키마(예시)
- item_id: str — 상품 식별자(ASIN/라벨). dedup 키로 사용
- title: str — 관찰에서 추출한 제목/요약
- price: str|float (optional) — 가격 텍스트 또는 숫자
- url: str|None — 현재 페이지 URL
- page: int|None — 검색 결과 페이지 번호
- keywords: list[str] — 해당 시점 검색에 사용한 키워드(item_config 기반)
- snapshot_excerpt: str — 관찰(obs)에서 앞부분 N자 스냅샷(예: 500자)
- rationale: str — 당시 선택/유망 판단 근거(thought)
- features_summary: str (optional) — Item 서브페이지 탐색에서 요약한 특징
- reviews_summary: str (optional) — 후기 요약
- actions_taken: list[str] — 해당 후보에 대해 수행한 클릭/서브페이지 열람 등 액션 시퀀스
- source_state: "Result"|"Item" — 기록 발생 상태
- score: float (optional) — 매칭 점수(휴리스틱/LLM 점수)
- last_seen_step: int — 마지막으로 업데이트된 step_count
- times_seen: int — 관찰/업데이트 누적 횟수

기록 포맷(JSON 예시)
```json
{
  "item_id": "B01ABCD",
  "title": "Logitech M185 Wireless Mouse",
  "price": "$12.99",
  "url": "http://127.0.0.1:3000/item_page/...",
  "page": 2,
  "keywords": ["wireless", "mouse", "budget"],
  "snapshot_excerpt": "[button] < Prev [button_]\nTitle: Logitech M185 ...",
  "rationale": "가격과 무선 조건을 충족하며 리뷰 평이 무난함",
  "features_summary": "콤팩트, 2.4GHz, 나노 리시버",
  "reviews_summary": "평균 4.3/5, 가격 대비 만족",
  "actions_taken": ["click[B01ABCD]", "click[features]", "click[< Prev]"],
  "source_state": "Item",
  "score": 0.72,
  "last_seen_step": 7,
  "times_seen": 2
}
```

기록 시점과 업데이트 규칙
- Result → Item 전이 시: 아이템 클릭 직전/직후에 최소 스냅샷을 기록(title/price/snapshot_excerpt/rationale/keywords/source_state="Result")
- Item 순환(서브페이지 열람) 후 Prev로 돌아갈 때: 동일 item_id 엔트리를 찾아 features_summary/reviews_summary/actions_taken/score 등을 병합 업데이트(last_seen_step++, times_seen++)
- 중복 처리: item_id 기준으로 존재하면 필드 업데이트, 없으면 append. 제목/가격이 비어있다면 후속 관찰로 보강
- 용량 제한: memory_buffer 최대 길이를 20~50 등으로 제한하고 LRU/저점수 우선 제거 정책 적용 권장

Stopping(백업 전략)에서의 선택 휴리스틱(예시)
- 1차: score가 있는 경우 최고 점수 선택
- 2차: 없는 경우 가격/키워드 매칭 휴리스틱으로 스코어링
- 3차: 마지막으로 본(last_seen_step) 또는 times_seen이 큰 후보 우선

참고: baseline의 browsed_items는 {item_id: [요약라인1, 가격라인, 마지막_rationale]} 형태였습니다. 위 스키마는 이를 일반화하여 다양한 정보와 업데이트 정책을 담을 수 있게 합니다.

간단 유틸 함수 예시
```python
def add_or_update_buffer(state: LaserState, candidate: dict) -> None:
    buf = state.get("memory_buffer", [])
    idx = next((i for i, c in enumerate(buf) if c.get("item_id") == candidate.get("item_id")), None)
    candidate["last_seen_step"] = state["step_count"]
    candidate["times_seen"] = (buf[idx].get("times_seen", 0) + 1) if idx is not None else 1
    if idx is None:
        buf.append(candidate)
    else:
        merged = {**buf[idx], **{k: v for k, v in candidate.items() if v not in (None, "")}}
        # actions_taken 병합
        at = list(dict.fromkeys((buf[idx].get("actions_taken") or []) + (candidate.get("actions_taken") or [])))
        merged["actions_taken"] = at
        buf[idx] = merged
    state["memory_buffer"] = buf
```

Result/Item 노드에서의 사용 예시
```python
# Result → Item으로 갈 때(아이템 클릭 직전)
add_or_update_buffer(state, {
    "item_id": decision.get("arg"),
    "title": extract_title(state["obs"]),
    "price": extract_price(state["obs"]),
    "keywords": state.get("item_config", {}).get("keywords", []),
    "snapshot_excerpt": state["obs"][:500],
    "rationale": state.get("thought_history", [])[-1] if state.get("thought_history") else "",
    "source_state": "Result",
    "actions_taken": [f"click[{decision.get('arg')}]"],
})

# Item에서 서브페이지를 본 뒤 Prev로 돌아갈 때
add_or_update_buffer(state, {
    "item_id": current_item_id,
    "features_summary": summarize_features(state["obs"]),
    "reviews_summary": summarize_reviews(state["obs"]),
    "actions_taken": [state.get("last_action", "")],
    "source_state": "Item",
})
```

### 4. 구현 가이드(실전)

아래는 WebShop 텍스트 환경(`WebAgentTextEnv`)과 연동되는 LangGraph 기반 LASER 에이전트의 최소 동작 스켈레톤입니다. 상태 정의, 노드 구현, 조건부 전환, 메모리 버퍼, 강제 종료(백업 전략)까지 포함합니다.

설치(예):
```bash
pip install "langgraph>=0.2.0" "langchain-core>=0.2.0"
```

예제 파일: `laser_langgraph_agent.py`
```python
from __future__ import annotations
from typing import List, Literal, Optional, TypedDict, NotRequired, Dict, Any
from dataclasses import dataclass

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
from web_agent_site.engine.engine import END_BUTTON, NEXT_PAGE, PREV_PAGE, BACK_TO_SEARCH

# 1) State 정의
class LaserState(TypedDict):
   # 입력/맥락
   user_instruction: str

   # 환경 상호작용
   obs: str                    # 현재 관찰 (html/text)
   url: Optional[str]
   current_laser_state: Literal["Search", "Result", "Item", "Stopping"]

   # 이력/메모리
   step_count: int
   thought_history: NotRequired[List[str]]
   action_history: NotRequired[List[str]]
   memory_buffer: NotRequired[List[dict]]   # 후보 아이템이나 유망 링크

   # 노드가 결정한 전이 및 산출물
   route: NotRequired[Literal["to_result", "to_item", "to_search", "stay_result", "stay_item", "to_stop"]]
   last_action: NotRequired[str]            # env에 보낸 raw action 문자열
   selected_item: NotRequired[dict]         # 최종 아이템(Stopping 용)
   info: NotRequired[dict]

# 2) 정책/LLM 인터페이스(선택적으로 교체 가능)
@dataclass
class Policy:
   # 실제 구현에서는 LLM(function-calling)로 교체 가능. 여기선 간단한 휴리스틱 데모.
   def decide_search_keywords(self, instr: str, obs: str) -> List[str]:
       # 매우 단순한 키워드 추출(데모)
       return [w for w in instr.lower().split() if len(w) > 2][:3] or ["shopping"]

   def decide_on_results(self, instr: str, obs: str) -> Dict[str, Any]:
       # 결과 페이지에서의 간단한 의사결정: 상품(asin) 텍스트가 보이면 그걸 클릭, 없으면 다음 페이지
       # 실제 구현은 관찰(obs) 파싱 후 상품 링크/버튼 후보를 추출하고, LLM이 함수 호출로 선택
       lines = obs.lower().splitlines()
       asin = None
       for ln in lines:
           # [button] B01ABC... 또는 제품 링크 라벨이 포함되어 있는지 헐겁게 탐지
           if "[button]" in ln and len(ln.strip()) > 20 and any(ch.isalnum() for ch in ln):
               asin = ln.replace('[button]', '').replace('[button_]', '').strip()
               break
       if asin:
           return {"action": "click", "arg": asin, "route": "to_item"}
       return {"action": "click", "arg": NEXT_PAGE.lower(), "route": "stay_result"}

   def decide_on_item(self, instr: str, obs: str) -> Dict[str, Any]:
       # 아이템 페이지에서 간단한 정책: 설명/후기 확인 1회 후 구매하거나 이전
       if "[button] buy now [button_]" in obs.lower():
           return {"action": "click", "arg": END_BUTTON.lower(), "route": "to_stop"}
       for candidate in ["description", "features", "reviews"]:
           if f"[button] {candidate} [button_]" in obs.lower():
               return {"action": "click", "arg": candidate, "route": "stay_item"}
       return {"action": "click", "arg": PREV_PAGE.lower(), "route": "to_result"}

# 3) 환경 래퍼 유틸
class EnvHandle:
   def __init__(self, observation_mode: str = "html"):
       self.env = WebAgentTextEnv(observation_mode=observation_mode)

   def reset(self, instruction_text: Optional[str] = None):
       obs, _ = self.env.reset(instruction_text=instruction_text)
       return self.env.observation, self.env.state.get("url")

   def step(self, raw_action: str):
       obs, reward, done, info = self.env.step(raw_action)
       url = self.env.state.get("url")
       return obs, url, reward, done, info

# 4) 노드 구현
policy = Policy()

def search_node(state: LaserState) -> dict:
   instr = state["user_instruction"]
   obs = state["obs"]
   keywords = policy.decide_search_keywords(instr, obs)
   raw = f"search[{ ' '.join(keywords) }]"
   obs2, url2, reward, done, info = state["_env"].step(raw)
   updates = {
       "obs": obs2,
       "url": url2,
       "last_action": raw,
       "step_count": state["step_count"] + 1,
       "action_history": state.get("action_history", []) + [raw],
       "current_laser_state": "Result",
       "route": "to_result",
       "info": {"reward": reward, "done": done} if info is None else info,
   }
   return updates

def result_node(state: LaserState) -> dict:
   instr, obs = state["user_instruction"], state["obs"]
   decision = policy.decide_on_results(instr, obs)
   raw = f"click[{decision['arg']}]" if decision["action"] == "click" else ""
   obs2, url2, reward, done, info = state["_env"].step(raw)

   # 간단한 메모리 버퍼 업데이트(예: 유망 아이템 문자열 스냅)
   mem = state.get("memory_buffer", [])
   if decision.get("route") == "to_item":
       mem = mem + [{"snapshot": obs[:500], "via": raw}]

   next_state = "Item" if decision["route"] == "to_item" else "Result"

   return {
       "obs": obs2,
       "url": url2,
       "last_action": raw,
       "step_count": state["step_count"] + 1,
       "action_history": state.get("action_history", []) + [raw],
       "memory_buffer": mem,
       "current_laser_state": next_state,
       "route": "to_item" if next_state == "Item" else "stay_result",
       "info": {"reward": reward, "done": done} if info is None else info,
   }

def item_node(state: LaserState) -> dict:
   instr, obs = state["user_instruction"], state["obs"]
   decision = policy.decide_on_item(instr, obs)
   raw = f"click[{decision['arg']}]"
   obs2, url2, reward, done, info = state["_env"].step(raw)
   route = decision["route"]

   updates: Dict[str, Any] = {
       "obs": obs2,
       "url": url2,
       "last_action": raw,
       "step_count": state["step_count"] + 1,
       "action_history": state.get("action_history", []) + [raw],
       "current_laser_state": "Stopping" if route == "to_stop" else ("Result" if route == "to_result" else "Item"),
       "route": route,
       "info": {"reward": reward, "done": done} if info is None else info,
   }
   return updates

def stopping_node(state: LaserState) -> dict:
   # 최종 산출물 구성. 백업 전략: mem 중 하나를 선택
   selected = state.get("selected_item")
   if not selected:
       mem = state.get("memory_buffer", [])
       selected = mem[-1] if mem else {"note": "no candidate"}
   return {"selected_item": selected}

# 5) 전이 라우터(조건부 엣지)
MAX_STEPS = 15

def router_fn(state: LaserState) -> str:
   if state["step_count"] >= MAX_STEPS:
       return "to_stop"
   route = state.get("route")
   if route in {"to_stop", "to_result", "to_item", "to_search", "stay_result", "stay_item"}:
       return route
   return "to_stop"

# 6) 그래프 구성

def build_graph():
   g = StateGraph(LaserState)
   g.add_node("Search", search_node)
   g.add_node("Result", result_node)
   g.add_node("Item", item_node)
   g.add_node("Stopping", stopping_node)

   g.add_edge(START, "Search")

   # Search -> Result 고정
   g.add_edge("Search", "Result")

   # Result 분기
   g.add_conditional_edges(
       "Result",
       router_fn,
       {
           "to_item": "Item",
           "stay_result": "Result",
           "to_stop": "Stopping",
       },
   )

   # Item 분기
   g.add_conditional_edges(
       "Item",
       router_fn,
       {
           "stay_item": "Item",
           "to_result": "Result",
           "to_stop": "Stopping",
       },
   )

   g.add_edge("Stopping", END)

   # 체크포인팅(옵션)
   memory = MemorySaver()
   return g.compile(checkpointer=memory)

# 7) 실행 유틸

def run_episode(instruction: Optional[str] = None, observation_mode: str = "html"):
   envh = EnvHandle(observation_mode=observation_mode)
   obs, url = envh.reset(instruction_text=instruction)

   graph = build_graph()
   init_state: LaserState = {
       "user_instruction": instruction or "Find a budget wireless mouse under $20.",
       "obs": obs,
       "url": url,
       "current_laser_state": "Search",
       "step_count": 0,
       "_env": envh,  # 내부 핸들러 참조를 상태에 담아 노드에서 사용
   }

   # 스트리밍으로 디버깅
   for event in graph.stream(init_state):
       print(event)

   final_state = graph.invoke(init_state)
   print("\n[FINAL] selected_item:", final_state.get("selected_item"))

if __name__ == "__main__":
   run_episode()
```

핵심 포인트
- 상태에 환경 핸들러(`_env`)를 담아 노드가 직접 `env.step`을 호출합니다. 실제 서비스에서는 외부 의존 분리를 위해 주입 방식/포트-어댑터 패턴을 권장합니다.
- `router_fn`으로 조건부 전환을 명시합니다. `MAX_STEPS`에 도달하면 강제 `Stopping`으로 전환해 백업 전략을 실행합니다.
- 현재 예제의 `Policy`는 휴리스틱이며, 실제 구현에서는 LLM(function-calling)로 교체하세요.

LLM 연동(개요)
- LangChain Runnable(ChatOpenAI 등)로 시스템 프롬프트에 상태별 지침을 넣고, 함수 정의로 허용 액션을 제공합니다.
- 모델이 반환한 함수 호출(name, arguments)에 따라 `search[...]`/`click[...]` 액션을 구성해 `env.step`을 호출합니다.
- 각 노드에서 `thought`를 기록하면 `thought_history` 기반의 추론 연속성이 향상됩니다.

### 5. 오프라인/리플레이 하네스 연동 팁
- `offline_replay_harness.md`의 스펙을 참고해, 위 그래프의 `EnvHandle`을 오프라인 환경 래퍼로 대체하면 무작위성 없는 재실행이 가능합니다.
- 에피소드 스냅샷(JSON)에서 상태 델타를 `graph.stream`으로 재생하면서 노드별 결정 로깅을 비교하면 리그레션 테스트가 수월합니다.

### 6. 개발 고려사항 및 추가 개선

*   **오류 처리 및 복구**: `langgraph`의 조건부 전환을 활용해 예외 상황에서 이전 상태로 되돌리거나 대안 액션을 시도하게 하세요. 네트워크/파싱 오류 시 `route="to_result"` 등으로 완만하게 복구.
*   **성능 모니터링**: `step_count`, `action_history` 기록으로 궤적 길이, 재방문 비율, 성공률을 추적합니다. `graph.stream`을 로깅 파이프라인과 연결하면 디버깅이 수월합니다.
*   **실제 배포 시 주의**: 실제 상용 웹에 적용 시 위험 액션(구매 등)은 인간 검증 단계를 넣으세요.
*   **확장성**: 계층적 에이전트, 추가 도구(지식 검색/계산기), 병렬 검색, 체크포인트 저장소 교체(Redis 등)를 통해 스케일업할 수 있습니다.
