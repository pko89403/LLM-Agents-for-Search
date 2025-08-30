# Knowledge-based Agent

LangGraph를 사용한 지식 기반 질의응답 에이전트 구현

## 🎯 프로젝트 개요

이 프로젝트는 사용자의 질문에 대해 웹 검색과 페이지 크롤링을 통해 정보를 수집하고, 수집된 지식을 바탕으로 정확한 답변을 제공하는 AI 에이전트입니다.

### 주요 기능
- **지능형 검색**: 사용자 질문에 맞는 최적 검색어 생성
- **웹 크롤링**: 검색 결과 페이지에서 관련 정보 추출
- **지식 통합**: 여러 소스의 정보를 종합하여 일관된 답변 생성
- **LangGraph 워크플로우**: 상태 기반 에이전트 실행 관리

## 🏗️ 아키텍처

```
├── main.py           # CLI 실행 스크립트
├── graph.py          # LangGraph 워크플로우 정의
├── nodes.py          # 에이전트 노드 구현
├── state.py          # 상태 및 데이터 모델
├── tools.py          # 외부 도구 (검색, 크롤링)
├── llm_utils.py      # LLM 관리 유틸리티
├── prompt.py         # 프롬프트 템플릿
├── prompt_utils.py   # 프롬프트 유틸리티
└── scripts/
    └── run_dev.sh    # 개발 환경 실행 스크립트
```

## 🚀 설치 및 실행

### 1. 의존성 설치
```bash
# uv 사용 (권장)
uv sync

# 또는 pip 사용
pip install -e .
```

### 2. 환경변수 설정
```bash
cp .env.example .env
# .env 파일을 편집하여 LLM 설정 추가
```

### 3. 실행 방법

#### 대화형 모드
```bash
python main.py --interactive
```

#### 단일 질문 모드
```bash
python main.py --goal "2024년 노벨 물리학상 수상자에 대해 알려주세요"
```

#### 데모 모드
```bash
python main.py
```

#### 개발 스크립트 사용
```bash
./scripts/run_dev.sh
```

## 📋 사용 예시

```python
from graph import run_knowledge_agent

# 질문 설정
goal = "2024년 노벨 물리학상 수상자에 대해 알려주세요"

# 에이전트 실행
result = run_knowledge_agent(goal, max_steps=10)

print(f"성공: {result['success']}")
print(f"스텝 수: {result['step_count']}")
print(f"도구 호출: {result['tool_calls']}회")
print(f"최종 답변: {result['final_answer']}")
```

## 🎛️ 설정

### 환경변수 (.env)
```bash
# LLM 설정
LLM_PROVIDER=ollama          # 또는 openai
OPENAI_API_KEY=your_key      # OpenAI 사용시
OPENAI_MODEL=gpt-4o-mini
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
```

### 주요 매개변수
- `--max-steps`: 최대 실행 스텝 수 (기본값: 10)
- `--interactive`: 대화형 모드 활성화
- `--goal`: 단일 질문 실행

### 지원 LLM
- **OpenAI**: GPT-4, GPT-3.5 등
- **Ollama**: 로컬 LLM (Gemma, Llama 등)

## 🔧 확장 가능성

- **다양한 도구 추가**: 뉴스 API, 학술 논문 검색 등
- **멀티모달 지원**: 이미지, 비디오 정보 처리
- **메모리 시스템**: 장기 기억 및 학습 능력
- **협업 에이전트**: 여러 전문 에이전트 협력

## 📊 평가 지표

- **성공률**: 질문에 대한 정확한 답변 제공율
- **효율성**: 평균 스텝 수 및 도구 호출 횟수
- **정보 품질**: 수집된 정보의 관련성 및 신뢰성
- **응답 완성도**: 답변의 포괄성 및 구체성

## 🛠️ 개발 가이드라인

프로젝트의 개발 가이드라인은 `.agent.md` 파일을 참조하세요.

### 주요 의존성
- **LangGraph**: 에이전트 워크플로우 관리
- **LangChain**: LLM 통합 및 도구 체인
- **Requests**: HTTP 요청 처리
- **BeautifulSoup4**: HTML 파싱 및 웹 크롤링
- **Pydantic**: 데이터 검증 및 모델링
- **Python-dotenv**: 환경변수 관리

### 개발 환경 설정
```bash
# 개발 의존성 설치
uv sync --dev

# 코드 포맷팅
black .

# 타입 체크
mypy .
```

## 🎪 데모 질문 예시

프로젝트에 포함된 데모 질문들:
- "2024년 노벨 물리학상 수상자에 대해 알려주세요"
- "100달러 이하의 무선 이어폰을 추천해주세요"
- "파이썬에서 비동기 프로그래밍이란 무엇인가요?"
- "최근 AI 기술 동향에 대해 설명해주세요"

## 📄 라이선스

MIT License