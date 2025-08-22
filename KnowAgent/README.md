# LangGraph KnowAgent

이 프로젝트는 KnowAgent 프롬프트/예제를 LangGraph 기반으로 구현한 에이전트입니다. KnowAgent는 ReAct(Reasoning + Acting) 패턴을 따르며, 다음의 Action Knowledge Graph를 통해 질문에 대한 다단계 추론을 수행합니다.

- Start: (Search, Retrieve)
- Retrieve: (Retrieve, Search, Lookup, Finish)
- Search: (Search, Retrieve, Lookup, Finish)
- Lookup: (Lookup, Search, Retrieve, Finish)
- Finish: ()

**주요 특징:**

*   **순차적 LLM 추론:** LLM은 각 스텝에서 `ActionPath` (이전까지 수행된 전체 행동 시퀀스), `Thought` (현재 상황에 대한 추론), `Action` (수행할 행동의 유형 및 인자)를 순차적으로 생성합니다.
*   **모듈화된 구조:** LLM 유틸리티(`knowagent_llm_utils.py`)와 도구(`knowagent_tools.py`)가 명확하게 분리되어 있습니다.
*   **Ollama 기본 지원:** 로컬 Ollama 모델(기본 `gemma3:4b`)을 기본 LLM으로 사용하도록 설정되어 있습니다.
*   **스크래치패드 관리:** LLM 컨텍스트 창 오버플로를 방지하기 위해 스크래치패드 자동 잘라내기 기능이 구현되어 있습니다.
*   **구성 가능한 안전 장치:** 에이전트의 동작을 제어하는 다양한 제한(연속 검색, 자동 종료 스텝, 컨텍스트 길이)을 설정할 수 있습니다.
*   **상세 로깅:** 에이전트의 내부 동작을 추적하고 디버깅하기 위한 상세한 로깅이 제공됩니다.
*   **코드 스타일:** `isort`와 `black`을 사용하여 코드 일관성을 유지합니다.

## 파일 구성

-   `graph.py`: LangGraph 그래프 정의 및 에이전트 실행 로직
-   `nodes.py`: 에이전트의 핵심 의사결정 및 도구 실행 노드 구현
-   `prompt_utils.py`: LLM 프롬프트 구성 및 스크래치패드 관리 유틸리티
-   `llm_utils.py`: LLM 모델(OpenAI, Ollama 등) 초기화 및 관리 유틸리티
-   `tools.py`: 에이전트가 사용하는 외부 도구(Wikipedia, Bing Search 등) 구현
-   `state.py`: LangGraph의 `AgentState` 정의
-   `main.py`: CLI 실행 스크립트
-   `prompt.py`: KnowAgent 프롬프트 템플릿, Few-shot 예제
-   `docs/agent_behavior_explanation.md` / `docs/knowagent_implementation.md`: 참고 문서

## 설치

```bash
pip install -r requirements.txt
```

## 환경변수

*   `LLM_PROVIDER`: 사용할 LLM 제공자 (기본: `ollama`). `openai`로 설정 시 OpenAI 모델 사용.
*   `OLLAMA_MODEL`: Ollama 모델명 (기본: `gemma3:4b`). `ollama:` 접두사 없이 모델명만 지정.
*   `OLLAMA_BASE_URL`: Ollama 서버 URL (기본: `http://localhost:11434`).
*   `OPENAI_API_KEY`: OpenAI API 키 (OpenAI 모델 사용 시 필요).
*   `OPENAI_MODEL`: OpenAI 모델명 (기본: `gpt-4o-mini`).
*   `BING_API_KEY`: Bing Search API 키 (선택). 없으면 위키백과 OpenSearch로 폴백.
*   `QUESTION`: 질문을 환경변수로 전달할 때 사용 (선택).

## 실행 예시

```bash

# 기본 Ollama 모델(gemma3:4b)로 실행
python main.py "Who is Milhouse named after in The Simpsons?"

# 특정 Ollama 모델(예: mistral)로 실행
python main.py --model ollama:mistral "Were Pavel Urysohn and Leonid Levin known for the same type of work?"

# OpenAI 모델(gpt-4o-mini)로 실행 (OPENAI_API_KEY 필요)
export OPENAI_API_KEY=sk-...
python main.py --model gpt-4o-mini --temperature 0.0 --max-steps 12 --question "What is the capital of France?"

# 모든 옵션 사용 예시
python main.py \
    --question "Who is Milhouse named after in The Simpsons?" \
    --model ollama:gemma3:4b \
    --temperature 0.0 \
    --max-steps 15 \
    --max-consec-search 5 \
    --auto-finish-step 8 \
    --context-len 1500
```
