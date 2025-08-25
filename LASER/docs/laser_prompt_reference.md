# LASER Prompt Reference (WebShop + LangGraph)

이 문서는 laser_agent.py와 prompt_library.py를 기반으로, LASER 에이전트가 WebShop 환경에서 사용하는 상태별 도구(함수), 프롬프트 템플릿, 액션 매핑 규칙을 정리합니다. LangGraph 구현 시 상태별 시스템 프롬프트 및 tool-call 구성을 빠르게 참조할 수 있도록 설계했습니다.

구성
- 상태별 툴 정의 목록
- 상태별 시스템 프롬프트 템플릿 정리
- 모델 출력 → 환경 액션 문자열 매핑 규칙
- 구매 플로우 체인 예시 (Item 상태)

## 1) 상태별 툴 정의 목록

- Search 상태
  - Search (search_items)
    - name: "Search"
    - parameters: { keywords: string (required), max_price: string (optional) }
    - purpose: 키워드로 검색 수행

- Result 상태
  - select_item (click_item)
    - name: "select_item"
    - parameters: { item_id: string (required) }
  - next_page (next_page)
    - name: "Next"
    - parameters: {}
  - back_to_search (back_to_search)
    - name: "Back_to_Search"
    - parameters: {}

- Item 상태
  - description (description)
    - name: "Description"
    - parameters: {}
  - features (features)
    - name: "Features"
    - parameters: {}
  - reviews (reviews)
    - name: "Reviews"
    - parameters: {}
  - buy_now (buy_item / buy_item_final)
    - name: "Buy_Now"
    - parameters: {}
  - previous_page (previous_page)
    - name: "Prev"
    - parameters: {}

참고: 정의는 prompt_library.py의 객체(search_items, click_item, description, features, reviews, buy_item, buy_item_final, previous_page, next_page, back_to_search)를 그대로 사용합니다.

## 2) 상태별 시스템 프롬프트 템플릿

- 공통 의사결정: chat_zero_shot_indiv_prompt_gpt4
  - 시스템 메시지: "You are an intelligent %s assistant ... Current observation: %s ... generate a rationale ..."
  - 상태 세부 설정 페이로드:
    - web_shop_search_gpt4: Search 상태용. Instruction + "[button] Search" 버튼 설명, 히스토리 고려 안내.
    - web_shop_select_gpt4: Result 상태용. Back to Search/Next/아이템 목록/가격 등의 레이아웃 제공.
    - web_shop_verify_gpt4: Item 상태용. Back/Prev/Customization/Description/Features/Reviews/Buy Now + 세부영역 설명 포함.

- 보조/매핑: chat_zero_shot_mapping_action_prompt_gpt4
  - 모델이 직접 함수 호출을 반환하지 않을 때, rationale → function-call로 매핑 유도.

- 커스터마이즈 추출: chat_zero_shot_custom_prompt
  - Item 상태에서 구매 직전, 커스터마이즈 타입명을 쉼표로 추출.

- 품질 관리(옵션):
  - chat_zero_shot_feedback_prompt_gpt4: 매니저가 액션/추론에 피드백 제공
  - chat_zero_shot_manager_prompt_gpt4: 히스토리 + 현재 관찰 + 제안
  - chat_zero_shot_rethink_prompt_gpt4: 피드백 반영 재결정

원문은 prompt_library.py를 참조하세요.

## 3) 모델 출력 → 환경 액션 매핑

- Search 상태
  - Search({keywords}) → env.action: `search[{keywords}]`

- Result 상태
  - select_item({item_id}) → env.action: `click[{item_id}]`
  - Next() → env.action: `click[Next >]`
  - Back_to_Search() → env.action: `click[Back to Search]`

- Item 상태
  - Description() → env.action: `click[description]`
  - Features() → env.action: `click[features]`
  - Reviews() → env.action: `click[reviews]`
  - Prev() → env.action: `click[< Prev]`
  - Buy_Now():
    - 커스터마이즈 추출(chat_zero_shot_custom_prompt) → buy_item_final로 선택지 인자 구성
    - 각 선택지에 대해 `click[{value}]` 실행
    - 최종 구매: `click[Buy Now]`

주의: 실제 클릭 라벨은 WebAgentTextEnv의 text_rich 모드에서 `[button] ... [button_]` 사이의 소문자/정규화된 문자열을 정확히 사용해야 합니다.

## 4) 구매 플로우 체인 예시 (Item 상태)

1) 검증 프롬프트로 다음 액션 결정
- 템플릿: chat_zero_shot_indiv_prompt_gpt4 + web_shop_verify_gpt4
- 허용 툴: Description/Features/Reviews/Buy_Now/Prev
- 출력 예: `{"name": "Description", "arguments": {}}`
- 매핑: `click[description]` 실행 → 세부 정보 관찰 업데이트

2) 필요한 세부정보 확인 후 구매 결정
- 출력 예: `{"name": "Buy_Now", "arguments": {}}`

3) 커스터마이즈 타입 추출
- 템플릿: chat_zero_shot_custom_prompt
- 모델 출력 예: `"Color, Size"` 또는 `"None"`
- None이 아니면 buy_item_final의 parameters.properties/required에 동적으로 반영

4) 커스터마이즈 선택 실행
- buy_item_final 호출 결과의 arguments로 각 옵션 값 클릭
- 예: `{"Color": "Black", "Size": "M"}` → `click[Black]`, `click[M]`

5) 최종 구매
- `click[Buy Now]`

6) 실패/보류 시
- Prev()로 Result로 복귀하거나, Result에서 Next/Back_to_Search로 재탐색

## 5) LangGraph 적용 팁
- 상태마다 해당 템플릿과 툴 집합을 바운드하여 모델 호출
- 모델의 tool_call이 없을 때는 chat_zero_shot_mapping_action_prompt_gpt4로 보조 호출
- Thought(추론) 텍스트는 thought_history에 축적하여 일관성 유지
- memory_buffer에 유망 후보를 계속 기록하고, Stopping에서 백업 전략으로 활용

이 레퍼런스를 laser_agent_dev_guide.md와 함께 사용하면, 상태/툴/프롬프트가 서로 어떻게 연결되는지 빠르게 이해하고 LangGraph 구현으로 옮길 수 있습니다.
