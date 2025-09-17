### **LangGraph 기반 'ToolDoc Refiner' v2 설계 문서 (상세 아키텍처 반영)**

#### **1. 개요 (Overview)**

본 문서는 LangGraph를 사용하여 **도구 문서화 파이프라인(Tool Documentation Pipeline)**을 완전 자동화하는 'ToolDoc Refiner' 에이전트의 v2 아키텍처를 설계한다.

**목표**는 사람이 거의 개입하지 않고, 신규 도구나 업스트림 업데이트 발생 시 **[수집 → 요약 → 검증 → 자기개선 → 배포]** 사이클을 자동으로 수행하는 것이다. 최종 산출물은 에이전트 런타임이 즉시 로드하여 사용할 수 있는 버전 관리된 **'ToolCard' 아티팩트**이다.

#### **2. 상위 아키텍처 및 LangGraph 흐름**

```
(START: RawDoc)
      |
      v
[1. Parse & Extract] ----(파싱 실패)----> [Handle Error]
      |
 (ToolSpec 생성)
      |
      v
[2. Summarize & Optimize]
      |
 (Concise Instruction 생성)
      |
      v
[3. Validate in Sandbox] ----(검증 성공)----> [5. Publish Artifact] --> (END: ToolCard)
      |
 (검증 실패 & 재시도 가능)
      |
      v
[4. Analyze Failure & Refine] --(수정 불가)--> [Handle Error]
      |
 (수정 프롬프트 생성)
      |
      +----(재시도 루프)----> [2. Summarize & Optimize]
```

#### **3. LangGraph 상태 (State) 정의 (v2)**

v1 상태를 확장하여 파이프라인의 복잡한 흐름을 모두 관리할 수 있도록 재설계한다.

```python
from typing import TypedDict, List, Dict, Any

class ToolProcessorState(TypedDict):
    # 1. 수집 (Crawl/Load)
    raw_doc_blob: Dict[str, Any] # {'content': str, 'source': str, 'version_hint': str}

    # 2. 분해 (Parse/Extract)
    tool_spec: Dict[str, Any] # {'name', 'summary', 'inputs', 'outputs', 'preconditions', 'error_modes'}

    # 3. 요약 (Summarize)
    concise_instruction: str # LLM이 생성한 간결한 지침

    # 4. 검증 및 자기개선 (Validate & Self-Refine)
    validation_logs: List[str] # 샌드박스 실행/스모크 테스트 결과 로그
    is_valid: bool # 검증 성공 여부
    refinement_prompt: str # 자기개선을 위한 수정 프롬프트
    retry_count: int # 무한 루프 방지를 위한 재시도 횟수

    # 5. 배포 (Publish)
    final_tool_card: Dict[str, Any] # 최종 ToolCard 아티팩트 (JSON)
    artifact_path: str # 저장된 아티팩트의 경로
    new_version: str # 부여된 Semantic Version (e.g., "1.3.0")

    # 공통
    error_message: str # 최종 실패 시 기록될 에러 메시지
```

#### **4. 노드 (Nodes) 설계 (v2)**

1.  **`parse_and_extract` (분해 및 추출)**
    - **역할**: `raw_doc_blob`을 입력받아 구조를 분석하고, `ToolSpec` 스키마에 따라 핵심 정보를 추출한다.
    - **입력**: `state['raw_doc_blob']`
    - **출력**: `state['tool_spec']`
    - **구현 노트**: OpenAPI, JSON-Schema 등 정형 데이터는 직접 파싱하고, Docstring이나 README 같은 비정형 데이터는 LLM을 이용해 `ToolSpec` 형식으로 구조화한다.

2.  **`summarize_and_optimize` (요약 및 최적화)**
    - **역할**: `tool_spec`을 기반으로 LLM 에이전트가 이해하기 쉬운 "Concise Instruction"을 생성한다. **자기개선 루프에서 `refinement_prompt`가 주어지면, 이를 반영하여 기존 지침을 수정한다.**
    - **입력**: `state['tool_spec']`, `state['refinement_prompt']` (선택적)
    - **출력**: `state['concise_instruction']`
    - **구현 노트**: 중복/장황한 설명을 제거하는 규칙 기반 최적화와 LLM 프롬프트 템플릿을 함께 사용한다.

3.  **`validate_in_sandbox` (샌드박스 검증)**
    - **역할**: 생성된 `concise_instruction`을 기반으로 실제 도구를 호출하는 테스트를 수행한다.
    - **입력**: `state['concise_instruction']`, `state['tool_spec']`
    - **출력**: `state['is_valid']`, `state['validation_logs']`
    - **구현 노트**:
        - **건식 호출 (Dry-run)**: 실제 실행 없이 파라미터만 전달하여 호출 가능 여부 확인
        - **모의 응답 (Mocking)**: 외부 API 의존성을 제거하고 예상 응답을 시뮬레이션
        - **스모크 테스트**: 필수 파라미터, 타입 일치, 예상되는 실패 케이스(e.g., 인증 실패) 등을 검증하는 간단한 테스트 실행

4.  **`analyze_failure_and_refine` (실패 분석 및 자기개선)**
    - **역할**: `validation_logs`를 분석하여 실패 원인을 분류하고, `summarize_and_optimize` 노드가 이해할 수 있는 **수정 프롬프트(`refinement_prompt`)**를 생성한다.
    - **입력**: `state['validation_logs']`
    - **출력**: `state['refinement_prompt']`, `state['retry_count']` (1 증가)
    - **구현 노트**: LLM을 사용하여 "로그를 보니 'api_key' 파라미터 설명이 누락되어 401 에러가 발생했습니다. 'api_key' 파라미터 설명을 추가하도록 지침을 수정하세요."와 같은 구체적인 수정 지시를 생성한다.

5.  **`publish_artifact` (아티팩트 배포)**
    - **역할**: 검증이 완료된 도구 지침을 최종 `ToolCard` 아티팩트로 생성하고, 버저닝하여 지정된 위치(`toolcards/`)에 저장한다.
    - **입력**: `state['concise_instruction']`, `state['tool_spec']`
    - **출력**: `state['final_tool_card']`, `state['artifact_path']`, `state['new_version']`
    - **구현 노트**: `tool_spec`의 변경 사항을 기반으로 Semantic Version(major/minor/patch)을 결정하고, `Changelog.md`에 변경 내역을 자동으로 추가한다.

#### **5. 최종 산출물: ToolCard (Final Artifact: ToolCard)**

에이전트 런타임이 직접 사용하는 최종 결과물. 런타임 정책까지 포함한다.

```json
// toolcards/get_weather@1.3.0.json
{
  "version": "1.3.0",
  "spec": {
    "name": "get_weather",
    "summary": "Retrieves the current weather for a specified location.",
    "inputs": {
      "location": { "type": "string", "required": true, "description": "The city and state, e.g., 'San Francisco, CA'." },
      "unit": { "type": "string", "required": false, "description": "Temperature unit ('celsius' or 'fahrenheit')." }
    },
    "outputs": { "temperature": "number", "condition": "string" },
    "error_modes": { "401": "Invalid API key.", "404": "Location not found." }
  },
  "concise_instruction": "Call get_weather(location, unit) to find the weather. Requires 'location'. 'unit' is optional.",
  "usage_examples": [
      "get_weather(location='Paris, FR')",
      "get_weather(location='New York, NY', unit='fahrenheit')"
  ],
  "runtime_policy": {
    "retry_on_failure": true,
    "retry_attempts": 2,
    "fallback_tool": "web_search"
  }
}
```

#### **6. 런타임 통합 (Runtime Integration)**

- **동적 로딩**: 에이전트 런타임은 시작 시 `toolcards/` 디렉토리에서 최신 버전의 ToolCard들을 동적으로 로드하여 메모리에 등록한다.
- **실행 가이드**: LLM 에이전트는 `concise_instruction`과 `usage_examples`를 보고 도구 사용법을 학습한다.
- **강화된 에러 핸들링**: 도구 호출 실패 시, `runtime_policy`에 따라 재시도를 하거나 지정된 `fallback_tool`을 사용하는 등 보다 지능적인 대처가 가능하다.
