# 에이전트 동작 설명: 단계별 분석

이 문서는 KnowAgent가 HotpotQA 데이터셋의 질문을 어떻게 처리하는지 설명합니다. 에이전트의 의사결정 과정을 이해하기 위해 각 단계별 "생각(Thought)"과 "행동(Action)"을 분석합니다.

## 질문 (Question)

**Question:**`Sierra Marjory Boggess s best known for originating the Broadway role of a fictional Disney character who is the only princess to become what?`

(시에라 마조리 보게스는 어떤 유일한 공주가 되는 디즈니 캐릭터의 브로드웨이 역할을 맡은 것으로 가장 잘 알려져 있습니까?)

## 에이전트의 전략

에이전트의 목표는 `Start`, `Search`, `Retrieve`, `Lookup`, `Finish`와 같은 행동 그래프(Action Graph)를 탐색하여 질문에 답변하는 것입니다. 각 단계에서 에이전트는 "Thought" 프로세스를 통해 현재까지 수집된 정보를 바탕으로 다음 "Action"을 결정합니다. 이 로직은 `run_hotpotqa.py` 스크립트 내의 `get_agent`를 통해 생성된 에이전트 객체에 의해 수행됩니다.

---

## 단계별 분석 (Step-by-Step Breakdown)

### 1단계: 초기 정보 검색 (Retrieve)

***ActionPath:**`Start`

***Thought:** 에이전트는 첫 단계로 질문의 핵심 엔티티(entity)인 "Sierra Marjory Boggess"에 대한 일반 정보를 얻는 것이 최선이라고 판단합니다. 위키피디아에서 정보를 가져오도록 설계된 `Retrieve` Action을 선택합니다.

***Action:**`Retrieve[Sierra Marjory Boggess]`

***Observation:** 에이전트는 다음 정보를 얻습니다: "Sierra Marjory Boggess is an American actress, best known for her role as Ariel in the 2007 Broadway production of *The Little Mermaid*." (시에라 마조리 보게스는 2007년 브로드웨이 작품 '인어공주'에서 아리엘 역으로 가장 잘 알려진 미국 배우입니다.)

**설명:** 에이전트는 질문의 핵심 엔티티를 성공적으로 식별하고 관련 초기 컨텍스트를 검색했습니다.

### 2단계: 특정 세부 정보 조회 (Lookup)

***ActionPath:**`Start -> Retrieve[Sierra Marjory Boggess]`

***Thought:** 에이전트는 이제 시에라 보게스가 아리엘을 연기했다는 것을 압니다. 질문의 핵심이 "어떤 유일한 공주가 되는가?"라는 것을 이해하고, 이는 독특한 지위를 의미한다고 판단합니다. 에이전트는 `Lookup` Action을 사용하여 아리엘의 역할에 대한 더 구체적인 정보를 찾기로 결정합니다.

***Action:**`Lookup[Ariel’s role]`

***Observation:** 에이전트는 핵심 정보 "Ariel is the first African-American princess in Disney’s animated canon." (아리엘은 디즈니 애니메이션 정식 작품 중 최초의 아프리카계 미국인 공주입니다.)를 찾습니다.

**설명:** 에이전트는 검색 범위를 성공적으로 좁혔습니다. 광범위한 `Search` 대신 `Lookup`을 사용하여 이미 가지고 있는 정보의 컨텍스트 내에서 특정 세부 정보를 찾습니다. 하지만 이 관찰은 사실과 다릅니다. (라이브 액션 리메이크의 할리 베일리 캐스팅에 영향을 받았거나, 모델의 할루시네이션(hallucination)일 수 있습니다.)

### 3단계: 최종 답변 공식화 (Finish)

***ActionPath:**`Start -> Retrieve[Sierra Marjory Boggess] -> Lookup[Ariel’s role]`

***Thought:** 에이전트는 답을 찾았다고 판단합니다. 2단계의 관찰이 질문의 "어떤 유일한 공주가 되는가?" 부분을 직접적으로 해결한다고 생각하고, 태스크를 완료하기에 충분한 정보라고 결론 내립니다.

***Action:**`Finish[the first African-American princess]`

**설명:** 에이전트는 수집한 정보를 바탕으로 자신있게 답변을 제공합니다. `Finish` Action은 프로세스의 종료를 의미합니다.

---

## 최종 답변 및 결론

***에이전트 생성 답변:**`the first African-American princess`

***실제 정답 (Ground Truth):**`a mother to her own child`

***결과:**`Correct: false`

에이전트는 질문을 분석하고 답을 찾기 위해 논리적인 단계별 프로세스를 따랐지만, 최종 답변은 틀렸습니다. 오류는 **2단계**에서 `Lookup` 행동이 사실과 다른 "Observation"을 반환하면서 시작되었습니다.

이 예시는 에이전트의 성능이 각 단계에서 검색하는 정보의 정확성에 얼마나 크게 의존하는지를 보여줍니다. 에이전트의 추론 과정 자체는 합리적이었지만, 부정확한 데이터를 기반으로 잘못된 결론에 도달했습니다.
