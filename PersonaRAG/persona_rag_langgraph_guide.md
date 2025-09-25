# LangGraph로 PersonaRAG 구현하기: 고급 개발자 가이드 v2.5

## 1. 소개

**PersonaRAG**는 전문화된 다중 에이전트를 통해 RAG 시스템을 개인화하는 혁신적인 아키텍처입니다. 이 가이드는 Python 개발자가 **LangGraph**의 최신 기능과 모범 사례를 활용하여 PersonaRAG의 핵심 사상인 '인지 동적 적응'을 효과적으로 구현하는 방법을 안내합니다.

이 가이드는 모든 에이전트가 메시지를 공유하고 소비하는 **블랙보드(Blackboard) 아키텍처**를 중심으로, **조건부 라우팅**을 통해 동적인 워크플로우를 구축하는 데 초점을 맞춥니다.

## 2. 사전 준비

구현에 필요한 핵심 라이브러리들을 설치합니다. `psycopg2-binary`는 프로덕션 환경에서 `PostgresSaver` 사용 시 필요합니다.

```bash
pip install langgraph langchain langchain_core langchain_openai psycopg2-binary
```

## 3. 핵심 아키텍처: 블랙보드 모델

PersonaRAG의 `Global Message Pool`을 LangGraph의 `State`를 이용한 '블랙보드'로 구현합니다.

*   **블랙보드 (State)**: 모든 에이전트는 자신의 결과물을 '메시지' 형태로 블랙보드에 게시(post)합니다. 다른 에이전트들은 블랙보드에 게시된 메시지들을 읽어 자신의 작업에 필요한 정보를 얻습니다.
*   **지능형 메시지 (Intelligent Messages)**: 각 메시지는 내용(content)뿐만 아니라, 메시지의 출처, 목적, 점수 등을 담은 **메타데이터(metadata)**를 가집니다. 이 메타데이터는 동적 라우팅의 핵심 근거가 됩니다.
*   **상태 관리 (`add_messages`)**: LangGraph의 `Annotated[List[AnyMessage], add_messages]`를 사용하여, 노드가 메시지를 반환할 때마다 상태의 `pool` 리스트에 자동으로, 그리고 병렬 실행 환경에서도 안전하게(thread-safe) 추가되도록 합니다.

## 4. 단계별 구현 가이드

### 4.1. 상태 정의: 블랙보드 (`GState`)

`TypedDict`를 사용하여 블랙보드 역할을 할 상태 객체를 정의합니다.

#### **Global Message Pool 설계 패턴**
`pool` 필드는 `Annotated[List[AnyMessage], add_messages]`로 정의합니다. 이는 PersonaRAG 논문이 말한 "중앙 허브" 역할을 LangGraph에서 구현하는 가장 표준적인 방법으로, 여러 브랜치(에이전트)가 동시에 `pool`에 메시지를 추가해도 충돌 없이 안전하게 처리되도록 보장합니다.

### 4.2. 에이전트 노드 구현

#### **에이전트 출력 계약 (Agent Output Contract)**
모든 에이전트 노드는 일관된 라우팅과 분석을 위해 반드시 아래의 '출력 계약'을 따라야 합니다. 출력은 항상 `{"pool": [AIMessage(...)]}` 형태여야 하며, `AIMessage`의 `metadata`에는 표준 스키마를 적용합니다.

#### **메타데이터 스키마 표준**
*   `from`: 메시지를 생성한 노드/에이전트 이름
*   `stage`: 메시지의 목적/단계 (e.g., `"profile_update"`, `"retrieval_proposal"`)
*   `route`: (선택) 제안하는 다음 경로 또는 전략
*   `score`: (선택) 제안의 신뢰도 또는 품질 점수
*   `ts`: 타임스탬프

## 5. 고급 오케스트레이션: 감독자(Supervisor) 패턴

PersonaRAG의 에이전트들은 독립적으로 작동하며 협업합니다. 이러한 흐름을 가장 잘 모델링하는 방법은 '감독자(Supervisor)' 역할을 하는 중앙 라우터를 두는 것입니다.

### 5.1. 감독자/핸드오프(Handoff) 패턴 소개

감독자 노드는 매번 에이전트가 작업을 완료한 후 호출됩니다. 감독자는 블랙보드(`pool`)의 전체 상태(최신 메시지, 요약, 사용자 피드백 등)를 '인지'하고, 다음에 어떤 에이전트를 실행할지 동적으로 결정하여 작업을 '핸드오프(handoff)'합니다. 이는 논문에서 여러 에이전트가 협업하는 방식과 정확히 일치합니다.

### 5.2. 피드백 루프 연결 설계

피드백은 '인지 동적 적응'의 핵심 루프를 완성합니다. `Feedback Agent`는 사용자의 암묵적/명시적 피드백을 수집하여 `profile`과 `pool`을 업데이트하고, 이는 다음번 상호작용에서 `Ranking` 및 `Contextual Retrieval` 에이전트의 결정에 즉시 영향을 미칩니다.

#### **데이터 흐름도**
```
[Synthesize Agent] -> 최종 답변 제시 -> [User Interaction]
      ^												 | (클릭, 평가 등)
      |												 v
[Profile] <--- [Feedback Agent] <- 피드백 수집
   |
   | (다음 사이클에 영향)
   v
[Ranking/Contextual Agents]
```

## 6. Cognitive Dynamic Adaptation (적응 루프) 구현

'인지 동적 적응'은 단순한 라우팅을 넘어, 여러 가설을 동시에 탐색하고, 사람의 개입을 허용하며, 실시간 신호로 스스로를 튜닝하는 완전한 루프를 의미합니다.

### 6.1. 병렬 탐색(Fan-out) → 승자 선택(Fan-in)

여러 검색 전략을 동시에 실행하여 최상의 결과를 선택하는 패턴입니다. 이는 논문의 '검토 및 수정 루프'를 구현하는 강력한 방법입니다.

### 6.2. 최종 응답 전 휴먼-인-더-루프 (Human-in-the-Loop)

비용이 많이 들거나 위험도가 높은 단계(예: 외부 유료 API 호출, 최종 답변 생성) 직전에 워크플로우를 일시 중지시켜 운영자의 승인/수정을 기다리게 할 수 있습니다.

*   **구현 방법**: `graph.compile()` 시 `interrupt_before` 또는 `interrupt_after` 인자를 사용합니다.

### 6.3. 온라인 튜닝 규칙 (Online Tuning Rules)

실시간 사용자 신호를 바탕으로 시스템 파라미터를 동적으로 조절하는 정책 테이블입니다. 이는 PersonaRAG의 "실시간 조정" 요구사항을 구체화한 것입니다.

| 조건 (Condition) | 행동 (Action) | 저장 위치 (Storage) |
| :--- | :--- | :--- |
| 사용자가 초기 검색 후 쿼리를 수정함 | `contextual_retrieval` 에이전트의 가중치/점수 상향 | `profile` (다음 세션용) |
| 사용자가 랭킹 5위 문서를 클릭함 | `live_session` 에이전트의 재랭킹 가중치 상향 | `pool` 메타데이터 (현재 세션용) |
| '전문가' 수준 문서에 오래 머무름 | `knowledge_level`을 'expert'로 갱신 | `profile` |
| 사용자가 답변에 '싫어요'를 누름 | 해당 답변 생성에 사용된 `route` 전략의 점수 하향 | `profile` |
| 사용자가 간결한 답변을 선호함 | `prompt.variant`를 'concise_summary'로 변경 | `profile` |

## 7. 스테이징/개발 환경을 위한 빠른 시작
...(이하 생략)... 

## 8. 고급 패턴 및 운영 방안
...(이하 생략)... 

## 9. 결론
v2.5 가이드는 **병렬 탐색(Fan-out/Fan-in)**, **휴먼-인-더-루프**, **온라인 튜닝 규칙**과 같은 프로덕션급 패턴을 추가하여, PersonaRAG의 '인지 동적 적응' 루프를 완성하는 구체적인 설계안을 제시합니다. 이 가이드를 통해 개발자는 단순한 RAG를 넘어, 실제 사용자와 상호작용하며 스스로 발전하는 지능형 시스템을 구축할 수 있습니다.
