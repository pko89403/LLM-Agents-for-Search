# Inference-time Tree Search 에이전트 구현 가이드 (LangGraph + WebShop Light)

> 이 문서는 **WebShop(가벼운 쇼핑 시뮬레이션)** 을 실험 환경으로 사용해 ITTS(Inference‑time Tree Search) 에이전트를 구현·평가하는 방법을 설명합니다. **로컬 Node 웹앱**, **텍스트/HTML 관측 모드**, **소형 데이터셋(예: 1k 상품 프리뷰)** 전제를 따릅니다. Docker 불필요.

## 목차
- [1. ITTS 개요와 핵심 구성요소](#1-itts-개요와-핵심-구성요소)
- [2. LangGraph 상태·노드 흐름](#2-langgraph-상태노드-흐름)
- [3. 프론티어 우선순위 큐](#3-프론티어-우선순위-큐)
- [4. 가치함수(Value) 설계](#4-가치함수value-설계)
- [5. WebShop 전용 액션(Action) 세트](#5-webshop-전용-액션action-세트)
- [6. 전이(Transition) 구현: Text / HTML 모드](#6-전이transition-구현-text--html-모드)
- [7. 베스트‑퍼스트 탐색 루프](#7-베스트퍼스트-탐색-루프)
- [8. WebShop 통합](#8-webshop-통합)
  - [8.1 설치·실행(소형 데이터셋)](#81-설치실행소형-데이터셋)
  - [8.2 관측 스키마(Observation)](#82-관측-스키마observation)
  - [8.3 어댑터 인터페이스(Env ↔ ITTS)](#83-어댑터-인터페이스env--itts)
  - [8.4 최소 실행 스니펫(Text 모드)](#84-최소-실행-스니펫text-모드)
- [9. 하이퍼파라미터·성능/안정성 트레이드오프](#9-하이퍼파라미터성능안정성-트레이드오프)
- [10. 실패 케이스 디버깅](#10-실패-케이스-디버깅)
- [11. 확장: 멀티 에이전트/다중 가치함수/빔서치](#11-확장-멀티-에이전트다중-가치함수빔서치)
- [12. 평가(쇼핑 과제 전용 지표)](#12-평가쇼핑-과제-전용-지표)
- [13. 배포 체크리스트](#13-배포-체크리스트)
- [14. 프로젝트 파일 구조 & 환경변수 패턴](#14-프로젝트-파일-구조--환경변수-패턴)
- [original.md 매핑](#originalmd-매핑)

## 1. ITTS 개요와 핵심 구성요소

Inference‑time Tree Search는 추론 시점에 **여러 경로를 분기·평가**해 최적의 다음 행동을 선택하는 방식입니다. 
- **State**: 현재 관측(텍스트/HTML), 행동 이력, 쿼리/필터, 장바구니, 페이지네이션 등.
- **Action**: 검색/필터/정렬/페이지 이동/상품 상세 열기/장바구니 담기 등 쇼핑 도메인에 특화.
- **Observation**: 결과 리스트(제목/가격/평점/ID), 현재 필터/정렬/페이지, 상세 페이지 정보.
- **Frontier**: 확장 대기 상태들의 우선순위 큐.
- **Value Function**: 상태의 목표 달성 가능성(예: 조건 일치 상품 발견 가능성)을 0~1로 점수화.
- **Backtracking**: 낮은 가치 분기는 중단하고 다른 분기로 전환.

## 2. LangGraph 상태·노드 흐름

```python
from typing import Any, List, Optional, Dict, TypedDict

class SearchState(TypedDict):
    goal: str
    observation: Dict[str, Any]    # results/items, filters, sort, page, cart, html 등
    action_history: List[str]
    frontier: List[tuple]
    best_state: Any
    best_score: float
    search_counter: int
    done: bool
```
- **초기화 노드**: 초기 관측 수집(홈/빈 검색) 및 1차 평가.
- **확장 노드**: 프론티어 팝 → 행동 후보 생성 → 전이 → 재평가 → 프론티어 푸시.
- **종료 노드**: 목표 달성/예산 소진 검사.

## 3. 프론티어 우선순위 큐

```python
import heapq
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class PrioritizedItem:
    priority: float
    state: Any = field(compare=False)

class Frontier:
    def __init__(self):
        self._heap = []
    def push(self, score: float, state: Any):
        heapq.heappush(self._heap, PrioritizedItem(-score, state))  # max-heap 효과
    def pop(self):
        if not self._heap:
            return None
        item = heapq.heappop(self._heap)
        return item.state, -item.priority
    def __len__(self):
        return len(self._heap)
```

## 4. 가치함수(Value) 설계

- 기준: **목표 적합성**(요구 스펙에 맞는 상품이 보이는가), **진행 가능성**(추가 필터/정제 행동이 명확한가), **리스크**(루프/막힘).
- LLM 스코어링 예시:
```python
VALUE_PROMPT = """
당신은 쇼핑 에이전트의 상태를 0~1로 평가합니다.
목표: {goal}
관측 요약: {observation}
행동 이력: {action_history}
한 줄로 0~1의 실수만 출력하세요.
"""
```

## 5. WebShop 전용 액션(Action) 세트

```python
from enum import Enum
from dataclasses import dataclass

class ActType(str, Enum):
    SEARCH = "search"              # 쿼리 문자열
    APPLY_FILTER = "apply_filter"  # (name, value)
    SORT_BY = "sort_by"            # e.g., price_asc, rating_desc
    CLICK_RESULT = "click_result"  # 결과 리스트의 idx
    OPEN_PRODUCT = "open_product"  # 상품 ID 직접 열기(선택)
    ADD_TO_CART = "add_to_cart"    # 현재 상세/리스트에서 ID/idx
    NEXT_PAGE = "next_page"
    PREV_PAGE = "prev_page"
    BACK = "back"
    HOME = "home"

@dataclass
class Action:
    type: ActType
    query: str | None = None
    name: str | None = None
    value: str | None = None
    sort_key: str | None = None
    idx: int | None = None
    product_id: str | None = None
```

- **권장 후보 생성 규칙**
  1) **검색이 비어있으면** `SEARCH(goal의 핵심 키워드)`
  2) 리스트가 있다면: `APPLY_FILTER`(가격상한/브랜드), `SORT_BY`(가격↑/평점↓), `CLICK_RESULT(상위 k)`
  3) 상세 페이지면: `ADD_TO_CART`, `BACK`
  4) 결과 부족 시: 쿼리 리라이트(동의어·속성 추가)

## 6. 전이(Transition) 구현: Text / HTML 모드

- **Text 모드(권장 초안)**: WebShop이 REST/JSON 엔드포인트로 **검색/필터/정렬/페이징**을 노출한다고 가정하고 `requests`로 호출 → JSON을 정규화하여 `observation` 구성.
- **HTML 모드**: `requests`(또는 Playwright)로 HTML을 가져와 **BeautifulSoup**으로 상품 카드(title/price/rating/id)와 UI 상태(필터/정렬/페이지)를 파싱.

```python
def transition(state, action: Action):
    if OBS_MODE == "text":
        # 예시: /api/search?q=...&sort=...&page=...&filter[name]=value
        obs = client.step_text(action)  # JSON → dict 정규화
    else:
        html = client.step_html(action) # HTML → parse → dict
        obs = parse_html_to_obs(html)
    return {
        **state,
        "observation": obs,
        "action_history": state["action_history"] + [str(action)],
        "done": goal_reached_by_obs(obs, state["goal"]),
    }
```

## 7. 베스트‑퍼스트 탐색 루프

```python
def best_first_search(initial_state, budget=64):
    frontier = Frontier()
    s0 = initial_state
    v0 = value_function(s0)
    frontier.push(v0, s0)

    best_state, best_score = s0, v0
    expanded = 0

    while len(frontier) > 0 and expanded < budget:
        item = frontier.pop()
        if item is None:
            break
        state, score = item
        if score > best_score:
            best_state, best_score = state, score
        if goal_reached(state):
            return state, score

        for a in propose_actions(state):
            ns = transition(state, a)
            nv = value_function(ns)
            frontier.push(nv, ns)
        expanded += 1
    return best_state, best_score
```

## 8. WebShop 통합

### 8.1 설치·실행(소형 데이터셋)
- **필수**: Node.js(≥18), npm, Python(≥3.10)
- **권장 절차(예시 스캐폴딩)**
```bash
# 1) WebShop 로컬 앱 준비
cd webshop
npm install
# 소형 데이터셋(1k) 사용: 환경변수 또는 설정 파일에서 경로 지정 (예: data/products_1k.jsonl)
export WEBSHOP_DATASET_PATH=./data/products_1k.jsonl
npm run dev   # 또는 npm start, 로컬 기본 포트 예: 3000

# 2) 에이전트 라이브러리(이 레포)
cd ../agent
pip install -r requirements.txt
```
> 실제 스크립트/이름은 사용하는 레포에 맞춰 조정하세요. 핵심은 **데이터셋 경량화(1k)** 와 **로컬 Node 앱 기동**입니다.

### 8.2 관측 스키마(Observation)
```python
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class Product:
    id: str
    title: str
    price: float
    rating: float | None = None
    brand: str | None = None
    attrs: Dict[str, Any] | None = None

@dataclass
class Observation:
    url: str | None
    query: str | None
    page: int
    sort: str | None
    filters: Dict[str, str]      # e.g., {"brand": "nike", "price_max": "100"}
    results: List[Product]
    cart: List[Product]
    html: str | None = None      # HTML 모드일 때만
```
- **요약자**: `observation → {top_k_titles, price_range, active_filters}`로 요약해 가치함수 입력 토큰을 줄입니다.

### 8.3 어댑터 인터페이스(Env ↔ ITTS)
```python
class WebShopClient:
    def reset(self) -> Observation: ...
    def search(self, q: str) -> Observation: ...
    def apply_filter(self, name: str, value: str) -> Observation: ...
    def sort_by(self, key: str) -> Observation: ...
    def click_result(self, idx: int) -> Observation: ...
    def add_to_cart(self, idx: int | None = None, product_id: str | None = None) -> Observation: ...
    def next_page(self) -> Observation: ...
    def prev_page(self) -> Observation: ...
    def back(self) -> Observation: ...
    def home(self) -> Observation: ...
```

### 8.4 최소 실행 스니펫(Text 모드)
```python
from typing import List

def propose_actions(state) -> List[Action]:
    obs = state["observation"]
    goal = state["goal"]
    actions = []
    if not obs.get("query"):
        actions.append(Action(type=ActType.SEARCH, query=extract_keywords(goal)))
        return actions
    # 후보: 필터/정렬/상위 k 클릭
    if need_price_cap(goal) and "price_max" not in obs["filters"]:
        actions.append(Action(type=ActType.APPLY_FILTER, name="price_max", value=str(extract_budget(goal))))
    actions.append(Action(type=ActType.SORT_BY, sort_key="price_asc"))
    k = min(3, len(obs["results"]))
    for i in range(k):
        actions.append(Action(type=ActType.CLICK_RESULT, idx=i))
    # 페이지 전환
    if len(obs["results"]) == 0:
        actions.append(Action(type=ActType.NEXT_PAGE))
    return actions

def goal_reached_by_obs(obs, goal: str) -> bool:
    # 예: "100달러 이하, 방수 러닝화" → 제목/속성/가격 조건 매칭
    for p in obs.get("results", []):
        if satisfies_goal(p, goal):
            return True
    return False
```

## 9. 하이퍼파라미터·성능/안정성 트레이드오프
- `budget`(확장 횟수), `branching`(행동 후보 수), LLM temperature, visited‑set(중복 상태 제거).
- 다양성 조절: ε‑greedy, 빔서치 혼합, 쿼리 리라이트 확률.

## 10. 실패 케이스 디버깅
- 로그: 현재 쿼리/필터/정렬/페이지, 상위 결과 요약, 선택 행동, 점수 기록.
- 리플레이: 동일 쿼리와 시드로 재현.
- 가드레일: 무진행 스텝 제한, 중복 쿼리 방지, 가격/속성 검증 실패 시 백트랙.

## 11. 확장: 멀티 에이전트/다중 가치함수/빔서치
- 가치함수 앙상블(평균/최대/가중합), 계획자‑실행자 분리, 룰‑기반 휴리스틱 결합.

## 12. 평가(쇼핑 과제 전용 지표)
- **성공률**: 목표 조건(가격 상한/속성/브랜드 등) 만족 상품 **발견·선택** 여부.
- **경로 길이/행동 수**: 효율성.
- **정확도(속성 일치율)**: 제목·속성 토큰 매칭, 가격 조건 위반 비율.
- **카트 정확도**: `ADD_TO_CART` 결과의 목표 적합성.

## 13. 배포 체크리스트
- 리소스: 모델 호출 한도/캐시.
- 안전: 허용 액션 제한, 타임아웃, 실패 시 폴백 쿼리.
- 관측성: 추적/알림/메트릭 대시보드.

## 14. 프로젝트 파일 구조 & 환경변수 패턴

### 14.1 표준 레이아웃(요약)
```text
project-root/
├─ agent/ (graph.py, nodes.py, state.py, value.py, actions.py)
├─ llm/   (client.py, prompt_utils.py, prompt.py, prompt_library.py)
├─ envs/webshop/ (client.py, parsing.py, replay.py)
├─ tools/ (web_search.py 등 선택)
├─ cli/   (main.py)
├─ configs/ (config.yaml, .env.example)
└─ scripts/ (run_dev.sh, run_eval.sh)
```

### 14.2 환경변수 스키마(.env.example)
```dotenv
# LLM
LLM_PROVIDER=ollama
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b

# WebShop
WEBSHOP_BASE_URL=http://localhost:3000
WEBSHOP_MODE=text           # text | html
WEBSHOP_DATASET_PATH=./data/products_1k.jsonl

# ITTS
ITTS_BUDGET=64
ITTS_BRANCHING=5
ITTS_SEED=42
LOG_DIR=./runs
```

### 14.3 `configs/config.yaml` 템플릿
```yaml
profiles:
  dev:
    env: webshop
    obs_mode: ${WEBSHOP_MODE:text}
    base_url: ${WEBSHOP_BASE_URL:http://localhost:3000}
    dataset_path: ${WEBSHOP_DATASET_PATH:./data/products_1k.jsonl}
    budget: ${ITTS_BUDGET:64}
    branching: ${ITTS_BRANCHING:5}
    log_dir: ${LOG_DIR:./runs}
  eval:
    env: webshop
    obs_mode: text
    budget: 128
    branching: 8
```

## original.md 매핑
- 프로젝트 개요 → 섹션 1
- 핵심 아키텍처 → 섹션 2, 7
- 에이전트 구현(`agent/`) → 섹션 4~7
- **환경 연동(WebShop 전용)** → 섹션 8
- 평가/운영(벤치마크, 배포) → 섹션 12~13

> **Note.** WebShop은 텍스트/HTML 중심 환경이라 **관측·액션 정의가 단순**하고, 소형 데이터셋으로 **빠른 ITTS 튜닝**이 가능합니다. 시각 요소(스크린샷/SoM)가 필요하다면 다른 환경과의 하이브리드 구성을 고려하세요.
