# AgentQ - Advanced AI Web Agent

🤖 **LangGraph 기반의 고급 웹 자동화 AI 에이전트**

AgentQ는 복잡한 웹 기반 작업을 자율적으로 수행하도록 설계된 AI 에이전트입니다. 사용자의 목표가 주어지면, AgentQ는 스스로 웹을 탐색하고 상호작용하여 과제를 해결합니다.

## 🎯 주요 특징

- **🧠 ReAct 기반 추론**: "Reason + Act" 사이클을 통해, 각 단계에서 추론하고 다음 행동을 결정합니다.
- **🤖 자율 실행 루프**: `THOUGHT` -> `ACTION` -> `STATUS` 의 간단하고 강력한 루프를 통해 작업을 수행합니다.
- **🌐 웹 자동화**: Playwright를 사용하여 최신 웹 기술과 상호작용합니다.
- **🔌 멀티 LLM 지원**: Ollama 로컬 모델 또는 OpenAI GPT 모델을 유연하게 사용할 수 있습니다.
- **🔧 간단하고 명확한 액션**: 사람이 이해하기 쉬운 몇 가지 기본 액션으로 웹을 제어합니다.

## 🏗️ 아키텍처

AgentQ는 LangGraph를 기반으로 한 간단한 ReAct(Reason+Act) 스타일의 루프를 사용합니다. 에이전트는 각 단계에서 다음 행동의 **이유(THOUGHT)**를 생각하고, **행동(ACTION)**을 결정한 뒤, 작업의 **상태(STATUS)**를 평가합니다.

```
START
  ↓
PLAN (최초 계획 수립)
  ↓
LOOP:
  THOUGHT (다음 행동 추론)
  ACTION (단일 행동 실행)
  CRITIQUE (진행 상태 및 완료 여부 평가)
  ↓
  ├─ CONTINUE → LOOP
  └─ COMPLETE → END
```

### 핵심 컴포넌트

- **Plan**: 전체 작업 계획을 처음에 한 번 수립합니다.
- **Thought & Action**: LLM이 현재 상태를 바탕으로 다음에 수행할 단일 행동과 그 이유를 생성합니다.
- **Critique**: 현재까지의 진행 상황을 평가하여 루프를 계속할지, 아니면 목표가 달성되어 종료할지를 결정합니다.

## 🛠️ 지원하는 액션

LLM은 다음의 간단한 형식으로 단일 액션을 결정합니다.

- `NAVIGATE: <url>`: 특정 URL로 이동합니다.
- `SEARCH: <search_query>`: Google에서 검색을 수행합니다.
- `CLICK: <css_selector>`: CSS 선택자로 특정 요소를 클릭합니다.
- `TYPE: <css_selector> || <text>`: 특정 요소에 텍스트를 입력합니다.
- `GET_DOM`: 현재 페이지의 DOM 정보를 가져옵니다.
- `WAIT: <seconds>`: 지정된 시간(초)만큼 대기합니다.
- `SCROLL: up|down`: 페이지를 위 또는 아래로 스크롤합니다.

## 📁 프로젝트 구조

```
agentq/                        # 핵심 AgentQ 모듈
├── state.py                   # LangGraph AgentState 정의
├── graph.py                   # LangGraph 그래프 정의 및 실행 로직
├── nodes.py                   # 핵심 의사결정 및 도구 실행 노드
├── prompt_utils.py            # 프롬프트 구성 및 스크래치패드 관리
├── llm_utils.py              # LLM 모델 초기화 및 관리
├── tools.py                  # 외부 도구 구현 및 웹 상호작용
├── prompt.py                 # 프롬프트 템플릿, Few-shot 예제
└── playwright_helper.py       # Playwright 브라우저 자동화

main.py                       # AgentQ 메인 실행 스크립트
requirements.txt              # Python 의존성
```

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. LLM 설정

#### Option A: OpenAI 사용
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

#### Option B: Ollama 사용 (권장)
```bash
# Ollama 설치 및 모델 다운로드 (예: llama3.2:3b)
ollama pull llama3.2:3b

# Ollama 서버 실행 (백그라운드에서 실행 추천)
ollama serve &
```

### 3. Chrome 디버깅 모드 시작

AgentQ는 실행 중인 Chrome 브라우저에 연결하여 동작합니다.

```bash
# 스크립트 실행 권한 부여
chmod +x scripts/setup_chrome.sh scripts/stop_chrome.sh

# Chrome 디버깅 모드 시작
./scripts/setup_chrome.sh
```

### 4. AgentQ 실행

#### 대화형 모드
```bash
python main.py
```

#### 단일 명령 실행
```bash
python main.py "네이버에서 오늘 날씨 검색해줘"
```

## 🔧 문제 해결

### Chrome 연결 실패
Chrome 브라우저가 포트 9222에서 실행되고 있는지 확인하세요.
```bash
# 기존 Chrome 프로세스 종료 후 재시작
./scripts/stop_chrome.sh
./scripts/setup_chrome.sh
```

### LLM 연결 실패
Ollama 서버가 실행 중인지 확인하세요.
```bash
# Ollama 상태 확인 (다른 터미널에서)
curl http://localhost:11434/api/tags

# 응답이 없으면 Ollama 서버 재시작
ollama serve &
```
