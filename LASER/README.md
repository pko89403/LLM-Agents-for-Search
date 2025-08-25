# LASER

## 🚀 실행 예제

### 기본 리플레이 모드 실행
```bash
# 세션 3번 리플레이 (기본)
python main.py --mode replay --session-id 3

# 피드백 시스템 활성화하여 실행
python main.py --mode replay --session-id 5 --enable-feedback

# 다른 LLM 모델 사용
python main.py --mode replay --session-id 10 --model gpt-4o-mini
```

### 실시간 모드 실행 (미구현)
```bash
# 실시간 WebShop 환경 (향후 구현 예정)
python main.py --mode real --instruction "Find a gaming laptop under $1500"
```

## 📁 파일 구성

### 🔧 핵심 에이전트 구현
-   `graph.py`: LangGraph 그래프 정의 및 에이전트 실행 로직 ✅
-   `nodes.py`: 에이전트의 핵심 의사결정 및 도구 실행 노드 구현 ✅
    - Search/Result/Item 상태 공간 노드
    - MANAGER 피드백 시스템
    - RETHINK 재고 시스템
    - 메모리 버퍼 관리
-   `state.py`: LangGraph의 `LaserState` 정의 ✅
-   `main.py`: CLI 실행 스크립트 (리플레이/실시간 모드 지원) ✅

### 🧠 LLM 및 프롬프트 시스템
-   `llm_utils.py`: LLM 모델(OpenAI, Ollama 등) 초기화 및 관리 유틸리티 ✅
-   `prompt_utils.py`: LLM 프롬프트 구성 및 스크래치패드 관리 유틸리티 ✅
    - 상태별 프롬프트 빌더
    - MANAGER/RETHINK 프롬프트 생성
    - 스크래치패드 토큰 관리
-   `prompt.py`: Agent 프롬프트 템플릿, Few-shot 예제 ✅
-   `prompt_library.py`: 원본 프롬프트 라이브러리 (참조용) ✅

### 🛠️ 도구 및 환경 연동
-   `tools.py`: 에이전트가 사용하는 WebShop 도구 구현 ✅
-   `tool_specs.py`: LLM Function Calling용 도구 명세 정의 ✅
-   `replay.py`: 오프라인 WebShop 리플레이 환경 ✅
-   `parsing_utils.py`: 웹페이지 관찰 파싱 유틸리티 ✅

### 🧪 데이터 및 설정
-   `webshop_demonstrations_0-100.json`: WebShop 데모 데이터 📊
-   `requirements.txt`: Python 의존성 목록 ⚙️
-   `.env.example`: 환경변수 설정 예제 ⚙️

### 📚 참조 및 문서
-   `docs/`: 상세 문서 디렉토리 📖

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

## 🚀 실행 예제

### 기본 리플레이 모드 실행
```bash
# 세션 3번 리플레이 (기본)
python main.py --mode replay --session-id 3

# 피드백 시스템 활성화하여 실행
python main.py --mode replay --session-id 5 --enable-feedback

# 다른 LLM 모델 사용
python main.py --mode replay --session-id 10 --model gpt-4o-mini
```

### 실시간 모드 실행 (미구현)
```bash
# 실시간 WebShop 환경 (향후 구현 예정)
python main.py --mode real --instruction "Find a gaming laptop under $1500"
```
