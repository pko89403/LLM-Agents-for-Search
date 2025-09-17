# CodeNav "Code-Use" 에이전트 LangGraph 구현 가이드 (v2)

## 1. 개요

이 문서는 **"CodeNav: Beyond tool-use to using real-world codebases with LLM agents"** 논문에서 제안된 **'code-use' 에이전트**를 LangGraph 기반으로 구현하기 위한 기술 가이드입니다.

기존의 절차적인 `Episode` 실행 루프를 LangGraph의 상태 기반 그래프 아키텍처로 전환하여, 에이전트의 상호작용 과정을 더 명확하게 모델링하고 모듈성과 확장성을 극대화하는 것을 목표로 합니다.

- **Tool-use vs. Code-use**: 이 구현은 미리 정의된 도구만 사용하는 'tool-use'를 넘어, 에이전트가 코드베이스를 직접 **검색(search)**하고, **실행(import/call)**하며 능동적으로 상호작용하는 'code-use' 패러다임을 따릅니다.

## 2. CodeNav 에이전트 상호작용 구조 (논문 기반)

논문에 기술된 에이전트의 상호작용 과정은 다음과 같이 요약할 수 있습니다.

1.  **입력**: 사용자의 자연어 쿼리(User Query)와 코드베이스의 고수준 설명(Library Description).
2.  **반복 과정 (Loop)**:
    -   **사고 (Thought)**: 현재까지의 정보를 바탕으로 다음 행동 계획을 수립합니다.
    -   **행동 (Action)**: 계획에 따라 아래 환경 중 하나와 상호작용합니다.
        -   **검색 (Retrieval Environment)**: Elasticsearch로 인덱싱된 코드베이스에서 관련 클래스, 함수, 코드 조각을 검색합니다.
        -   **실행 (Execution Environment)**: 검색된 코드를 `import` 하거나 직접 실행하여 결과를 확인하고, 변수 상태의 변화를 관찰합니다.
3.  **종료**: 사용자의 쿼리가 해결되었다고 판단하면 `done` 행동으로 과정을 마칩니다.

## 3. LangGraph 기반 신규 아키텍처 제안

논문의 상호작용 구조를 LangGraph로 모델링합니다. 역할에 따라 노드를 명확히 분리하여 아키텍처의 명확성을 높입니다.

#### 3.1. 그래프 상태 (State)

각 노드 간에 전달될 상태 객체입니다. 논문의 메소드를 반영하여 상세화합니다.

```python
# codenav/langgraph/state.py (신규 생성)

from typing import List, Literal, TypedDict, Set
from codenav.interaction.messages import Interaction, CodeNavAction, ExecutionResult, RetrievalResult

class CodeNavState(TypedDict):
    # 전체 상호작용의 히스토리
    interactions: List[Interaction]
    # 사용자의 최초 질문
    user_query: str
    # 코드베이스 고수준 설명
    repo_description: str
    # 가장 최근에 에이전트가 결정한 행동
    latest_action: CodeNavAction
    # 검색 시, 이미 반환된 결과를 추적하기 위한 집합
    retrieved_doc_ids: Set[str]
    # 코드 실행 환경의 전역 변수 상태
    global_vars: dict
```

#### 3.2. 그래프 노드 (Nodes)

1.  **`agent_node` (사고 및 계획 노드)**
    -   **역할**: 다음에 수행할 행동(`search`, `code`, `done`)을 결정합니다.
    -   **구현**:
        -   `interactions` 히스토리, `user_query`, `repo_description`을 조합하여 LLM 프롬프트를 생성합니다.
        -   LLM 호출 후, 응답 XML을 `CodeNavAction(thought, type, content)`으로 파싱하여 `latest_action` 상태를 업데이트합니다.

2.  **`retrieval_node` (검색 노드)**
    -   **역할**: `Retrieval Environment`와 상호작용하여 코드베이스를 검색합니다.
    -   **구현**:
        -   `latest_action`이 `search`일 때 호출됩니다.
        -   `RetrievalEnv.step(action)`을 실행합니다. 이 때, `retrieved_doc_ids`를 참조하여 이전에 반환된 문서는 제외하고 새로운 결과를 가져옵니다.
        -   반환된 `RetrievalResult`를 `interactions` 히스토리에 추가하고, 새로 검색된 문서 ID들을 `retrieved_doc_ids`에 추가합니다.
        -   **Response 포맷팅**: 논문에 따라, 상위 K개 결과는 소스코드/Docstring을, 나머지는 프로토타입(시그니처)만 보여주도록 `RetrievalResult.format()` 메소드를 수정/활용합니다.

3.  **`execution_node` (실행 노드)**
    -   **역할**: `Execution Environment`와 상호작용하여 코드를 실행합니다.
    -   **구현**:
        -   `latest_action`이 `code`일 때 호출됩니다.
        -   `PythonCodeEnv.step(action)`을 실행합니다. 이 때, `global_vars` 상태를 `exec` 함수의 `global` 컨텍스트로 전달합니다.
        -   실행 후, 반환된 `ExecutionResult`를 `interactions`에 추가하고, 변경된 `global_vars`를 다시 상태에 업데이트합니다.

#### 3.3. 그래프 흐름 (Edges)

```
                  (start)
                     |
                     v
             +----------------+
             |   agent_node   |  (계획 수립: search, code, done?)
             +----------------+
                     |
                     v
 (router: latest_action.type에 따라 분기)
   |
   +---(action == 'search')--->+------------------+
   |                          |  retrieval_node  |
   |                          +------------------+
   |                                   |
   +---(action == 'code')----->+------------------+
   |                          | execution_node   |
   |                          +------------------+
   |                                   |
   '---(action == 'done')----->[      END       ]
                                       |
             (두 노드 모두)------------+
                     |
                     v
                  (agent_node로 복귀)
```

## 4. 구현 단계 상세

1.  **의존성 추가**: `requirements.txt`에 `langgraph`를 추가합니다.

2.  **초기 상태 구성**: 에이전트 실행 시, `user_query`와 `repo_description`을 포함한 `CodeNavState`를 초기 입력으로 구성합니다. `global_vars`와 `retrieved_doc_ids`는 빈 상태로 시작합니다.

3.  **상태/노드/그래프 구현**: 위의 아키텍처 제안에 따라 `state.py`, `nodes.py`, `langgraph_run.py` 파일을 작성합니다.
    -   `execution_node`는 `global_vars`를 인자로 받고 업데이트된 `global_vars`를 반환하도록 시그니처를 수정해야 할 수 있습니다.

4.  **Environment 및 Message 수정/확인**:
    -   `RetrievalEnv`: `step` 메소드가 `retrieved_doc_ids`를 인자로 받아 중복을 제거하는 기능이 있는지 확인하고, 없다면 추가합니다.
    -   `RetrievalResult`: `format()` 메소드가 논문에서 제안된 '소스코드+프로토타입' 하이브리드 방식을 지원하는지 확인하고, 필요시 수정합니다.
    -   `PythonCodeEnv`: `step` 메소드가 외부에서 `global_vars`를 받아 사용하고, 실행 후 변경된 `global_vars`를 반환하는 구조인지 확인하고, 필요시 수정합니다.

## 5. 기대 효과

-   **명확한 아키텍처**: 논문의 '검색-실행' 루프가 그래프 구조에 명확하게 매핑되어 코드의 가독성과 이해도가 향상됩니다.
-   **디버깅 용이성**: LangSmith를 통해 각 노드(사고, 검색, 실행)의 입출력과 상태 변화를 시각적으로 추적할 수 있습니다.
-   **유연한 확장**: '코드 요약', '테스트 실행' 등 새로운 'code-use' 행동을 추가하고 싶을 때, 해당 `Environment`와 그래프 노드를 추가하는 방식으로 손쉽게 확장할 수 있습니다.
