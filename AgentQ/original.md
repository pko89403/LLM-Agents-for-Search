# AgentQ 프로젝트 쉐도잉 구현 가이드

## 목차
1. [프로젝트 개요](#프로젝트-개요)
2. [아키텍처 분석](#아키텍처-분석)
3. [핵심 컴포넌트](#핵심-컴포넌트)
4. [구현 단계별 가이드](#구현-단계별-가이드)
5. [기술 스택](#기술-스택)
6. [설정 및 환경 구성](#설정-및-환경-구성)
7. [코드 구조 분석](#코드-구조-분석)
8. [실습 과제](#실습-과제)

## 프로젝트 개요

AgentQ는 웹 환경에서 자율적으로 작업을 수행하는 고급 AI 에이전트입니다. 이 프로젝트는 다음과 같은 핵심 기능을 제공합니다:

### 주요 특징
- **다중 에이전트 아키텍처**: Planner-Navigator, Actor-Critic 구조
- **Monte Carlo Tree Search (MCTS)**: 강화학습 기반 의사결정
- **Direct Preference Optimization (DPO)**: 파인튜닝 기법
- **웹 자동화**: Playwright 기반 브라우저 제어
- **시각적 이해**: 스크린샷 및 DOM 분석

### 연구 배경
이 프로젝트는 "Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents" 논문의 오픈소스 구현체입니다.

## 아키텍처 분석

### 1. 전체 시스템 아키텍처

AgentQ는 다음과 같은 4가지 주요 아키텍처를 지원합니다:

#### 1.1 Planner-Navigator 멀티에이전트 아키텍처
```
User Input → Planner Agent → Task Planning → Navigator Agent → Web Actions → Result
```

#### 1.2 Solo Planner-Actor 에이전트
```
User Input → AgentQ Base → Planning + Action → Web Execution → Result
```

#### 1.3 Actor-Critic 멀티에이전트 아키텍처
```
User Input → Actor Agent → Proposed Actions → Critic Agent → Action Ranking → Execution
```

#### 1.4 MCTS + DPO 강화학습 아키텍처
```
State → MCTS Search → Action Selection → DPO Training → Policy Update
```

### 2. 상태 관리 시스템

시스템은 다음과 같은 상태들을 관리합니다:

```python
class State(Enum):
    PLAN = "plan"                    # 계획 수립 단계
    BROWSE = "browse"                # 웹 탐색 단계
    AGENTQ_BASE = "agentq_base"      # 기본 AgentQ 실행
    AGENTQ_ACTOR = "agentq_actor"    # Actor 에이전트 실행
    AGENTQ_CRITIC = "agentq_critic"  # Critic 에이전트 실행
    COMPLETED = "completed"          # 작업 완료
```

### 3. 메모리 구조

```python
class Memory:
    objective: str                    # 사용자 목표
    current_state: State             # 현재 상태
    plan: List[Task]                 # 계획된 작업들
    thought: str                     # 현재 사고 과정
    completed_tasks: List[Task]      # 완료된 작업들
    current_task: Optional[Task]     # 현재 진행 중인 작업
    final_response: Optional[str]    # 최종 응답
```

## 핵심 컴포넌트

### 1. Orchestrator (오케스트레이터)

시스템의 중앙 제어 장치로, 모든 에이전트와 상태를 관리합니다.

**주요 기능:**
- 상태 전환 관리
- 에이전트 간 통신 조율
- 메모리 업데이트
- 웹 드라이버 관리

**핵심 메서드:**
```python
async def execute_command(command: str)     # 명령 실행
async def _handle_state()                   # 상태별 처리
async def handle_agentq_actions()          # 액션 실행
```

### 2. Base Agent (기본 에이전트)

모든 에이전트의 부모 클래스입니다.

**주요 특징:**
- LLM 통신 관리
- 프롬프트 처리
- 입출력 형식 검증
- 추적 및 로깅

### 3. Skills (스킬 시스템)

웹 상호작용을 위한 기본 동작들을 정의합니다.

**주요 스킬들:**
- `click_using_selector`: 요소 클릭
- `enter_text_using_selector`: 텍스트 입력
- `get_dom_with_content_type`: DOM 정보 추출
- `get_screenshot`: 스크린샷 캡처
- `open_url`: URL 열기
- `solve_captcha`: 캡차 해결

### 4. Web Driver (웹 드라이버)

Playwright 기반의 브라우저 제어 시스템입니다.

**기능:**
- 브라우저 인스턴스 관리
- 페이지 네비게이션
- 스크린샷 캡처
- DOM 조작

### 5. MCTS (Monte Carlo Tree Search)

강화학습을 위한 트리 탐색 알고리즘입니다.

**구성 요소:**
- Node: 상태 노드
- Edge: 액션 엣지
- Search: 탐색 알고리즘
- Evaluation: 상태 평가

## 구현 단계별 가이드

### Phase 1: 기본 환경 설정 (1-2주)

#### 1.1 개발 환경 구성
```bash
# Poetry 설치
curl -sSL https://install.python-poetry.org | python3 -

# 프로젝트 클론 및 의존성 설치
git clone <your-repo>
cd agentq-shadow
poetry install

# 환경 변수 설정
cp .env.example .env
# OpenAI API 키, Langfuse 키 등 설정
```

#### 1.2 브라우저 설정
```bash
# Chrome 개발자 모드 실행
# macOS
sudo /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222

# Linux
google-chrome --remote-debugging-port=9222

# Windows
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222
```

#### 1.3 기본 구조 이해
- `agentq/core/`: 핵심 로직
- `agentq/core/agent/`: 에이전트 구현
- `agentq/core/skills/`: 웹 스킬
- `agentq/core/orchestrator/`: 오케스트레이터
- `test/`: 테스트 및 평가

### Phase 2: 기본 에이전트 구현 (2-3주)

#### 2.1 BaseAgent 클래스 구현
```python
class BaseAgent:
    def __init__(self, name, system_prompt, input_format, output_format):
        self.name = name
        self.system_prompt = system_prompt
        self.input_format = input_format
        self.output_format = output_format

    async def run(self, input_data, session_id):
        # LLM 호출 및 응답 처리
        pass
```

#### 2.2 AgentQ 기본 구현
```python
class AgentQ(BaseAgent):
    def __init__(self):
        self.ltm = self.__get_ltm()
        self.system_prompt = self.__modify_system_prompt(self.ltm)
        super().__init__(
            name="agentq",
            system_prompt=self.system_prompt,
            input_format=AgentQBaseInput,
            output_format=AgentQBaseOutput
        )
```

#### 2.3 스킬 시스템 구현
각 스킬을 독립적으로 구현:
- DOM 추출 스킬
- 클릭 스킬
- 텍스트 입력 스킬
- 네비게이션 스킬

### Phase 3: 오케스트레이터 구현 (2-3주)

#### 3.1 상태 관리 시스템
```python
class Orchestrator:
    def __init__(self, state_to_agent_map):
        self.state_to_agent_map = state_to_agent_map
        self.memory = None

    async def execute_command(self, command):
        self.memory = Memory(objective=command, current_state=State.AGENTQ_BASE)
        while self.memory.current_state != State.COMPLETED:
            await self._handle_state()
```

#### 3.2 액션 실행 시스템
```python
async def handle_agentq_actions(self, actions):
    results = []
    for action in actions:
        if action.type == ActionType.CLICK:
            result = await click(selector=f"[mmid='{action.mmid}']")
        elif action.type == ActionType.TYPE:
            result = await entertext(action.content, action.mmid)
        # ... 기타 액션 처리
        results.append(result)
    return results
```

### Phase 4: 멀티에이전트 시스템 (3-4주)

#### 4.1 Actor-Critic 아키텍처
```python
class AgentQActor(BaseAgent):
    # 여러 가능한 액션 제안

class AgentQCritic(BaseAgent):
    # 제안된 액션들을 평가하고 순위 매기기
```

#### 4.2 Planner-Navigator 아키텍처
```python
class PlannerAgent(BaseAgent):
    # 고수준 계획 수립

class BrowserNavAgent(BaseAgent):
    # 저수준 웹 네비게이션
```

### Phase 5: MCTS 및 강화학습 (4-5주)

#### 5.1 MCTS 구현
```python
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

class MCTS:
    def search(self, root_state, num_simulations):
        # UCB1 기반 탐색
        # 시뮬레이션 및 백프로파게이션
        pass
```

#### 5.2 DPO 훈련 시스템
```python
def generate_dpo_pairs():
    # MCTS를 통한 선호도 데이터 생성
    # 성공/실패 액션 쌍 생성
    pass
```

## 기술 스택

### 핵심 라이브러리
- **Python 3.10+**: 기본 언어
- **Playwright**: 웹 브라우저 자동화
- **OpenAI API**: LLM 통신
- **Pydantic**: 데이터 검증 및 직렬화
- **AsyncIO**: 비동기 프로그래밍
- **LiteLLM**: 다중 LLM 지원

### 의존성 관리
```toml
[project]
dependencies = [
    "litellm<2.0.0,>=1.42.9",
    "pydantic<3.0.0,>=2.8.2",
    "pytest-playwright<1.0.0,>=0.5.1",
    "playwright-stealth<2.0.0,>=1.0.6",
    "openai<2.0.0,>=1.40.1",
    "aiohttp<4.0.0,>=3.10.2",
    "langsmith<1.0.0,>=0.1.104",
    "instructor<2.0.0,>=1.4.0"
]
```

### 개발 도구
- **Poetry**: 패키지 관리
- **Ruff**: 코드 포맷팅 및 린팅
- **Pytest**: 테스트 프레임워크
- **Langfuse**: LLM 추적 및 모니터링

## 설정 및 환경 구성

### 환경 변수 설정
```bash
# .env 파일
OPENAI_API_KEY=your_openai_api_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com

# 선택적 설정
AGENTOPS_API_KEY=your_agentops_key
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
```

### 프로젝트 구조
```
agentq/
├── __init__.py
├── __main__.py
├── config/
│   └── config.py
├── core/
│   ├── agent/          # 에이전트 구현
│   ├── mcts/           # MCTS 알고리즘
│   ├── memory/         # 메모리 관리
│   ├── models/         # 데이터 모델
│   ├── orchestrator/   # 오케스트레이터
│   ├── prompts/        # 프롬프트 템플릿
│   ├── skills/         # 웹 스킬
│   └── web_driver/     # 브라우저 제어
├── utils/              # 유틸리티 함수
└── user_preferences/   # 사용자 설정
```

## 코드 구조 분석

### 1. 데이터 모델 (models.py)

#### 핵심 모델들
```python
class Task(BaseModel):
    id: int
    description: str
    url: Optional[str]
    result: Optional[str]

class Action(BaseModel):
    type: ActionType
    mmid: Optional[str]
    content: Optional[str]
    website: Optional[str]
    timeout: Optional[int]

class Memory(BaseModel):
    objective: str
    current_state: State
    plan: List[Task]
    completed_tasks: List[Task]
    final_response: Optional[str]
```

### 2. 프롬프트 시스템 (prompts.py)

#### 주요 프롬프트 구조
```python
LLM_PROMPTS = {
    "AGENTQ_BASE_PROMPT": """
    You are AgentQ, an advanced AI agent...
    Current objective: {objective}
    Current page DOM: {current_page_dom}
    """,

    "AGENTQ_ACTOR_PROMPT": """
    You are the Actor agent in AgentQ...
    Propose multiple possible actions...
    """,

    "AGENTQ_CRITIC_PROMPT": """
    You are the Critic agent in AgentQ...
    Evaluate and rank the proposed actions...
    """
}
```

### 3. 스킬 시스템 상세

#### DOM 추출 스킬
```python
async def get_dom_with_content_type(content_type: str):
    # content_type: "all_fields", "input_fields", "text_only"
    if content_type == "all_fields":
        return await do_get_accessibility_info(page, only_input_fields=False)
    elif content_type == "input_fields":
        return await do_get_accessibility_info(page, only_input_fields=True)
    elif content_type == "text_only":
        return await get_filtered_text_content(page)
```

#### 클릭 스킬
```python
async def click(selector: str, wait_before_execution: float = 1):
    page = await PlaywrightManager().get_current_page()
    await page.wait_for_selector(selector, timeout=10000)
    await page.click(selector)
    return f"Clicked on element with selector: {selector}"
```

### 4. 실행 흐름

#### 기본 AgentQ 실행 흐름
```
1. 사용자 입력 → Memory 초기화
2. DOM 추출 → AgentQ 에이전트 호출
3. 액션 계획 → 액션 실행
4. 결과 평가 → 다음 단계 결정
5. 완료 또는 반복
```

#### Actor-Critic 실행 흐름
```
1. Actor: 여러 액션 제안
2. Critic: 액션 평가 및 순위
3. 최고 액션 실행
4. 결과 피드백
5. 반복 학습
```

## 실습 과제

### 초급 과제 (1-2주)

#### 과제 1: 기본 스킬 구현
```python
# 목표: 간단한 웹 스킬 구현
async def my_custom_skill():
    # 1. 페이지 로드
    # 2. 특정 요소 찾기
    # 3. 상호작용 수행
    # 4. 결과 반환
    pass
```

#### 과제 2: 간단한 에이전트 만들기
```python
class SimpleAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="simple_agent",
            system_prompt="You are a simple web agent...",
            input_format=SimpleInput,
            output_format=SimpleOutput
        )
```

### 중급 과제 (2-3주)

#### 과제 3: 멀티스텝 태스크 구현
- 로그인 → 검색 → 결과 수집 파이프라인 구현
- 에러 핸들링 및 재시도 로직 추가

#### 과제 4: 커스텀 오케스트레이터
```python
class CustomOrchestrator(Orchestrator):
    async def _handle_custom_state(self):
        # 새로운 상태 처리 로직
        pass
```

### 고급 과제 (3-4주)

#### 과제 5: MCTS 기반 계획 시스템
```python
class PlanningMCTS:
    def __init__(self):
        self.root = None

    def plan_actions(self, current_state):
        # MCTS를 사용한 액션 계획
        pass
```

#### 과제 6: 강화학습 통합
- DPO 데이터 생성 파이프라인 구현
- 모델 파인튜닝 스크립트 작성

### 실전 프로젝트 (4-6주)

#### 프로젝트 1: E-commerce 자동화
- 제품 검색 및 비교
- 장바구니 추가
- 결제 프로세스 자동화

#### 프로젝트 2: 소셜 미디어 관리
- 포스트 작성 및 스케줄링
- 댓글 모니터링
- 분석 리포트 생성

## 평가 및 테스트

### 테스트 실행
```bash
# 기본 테스트
python -m test.tests_processor --orchestrator_type fsm

# MCTS 테스트
python -m agentq.core.mcts.browser_mcts

# 커스텀 테스트
python -m test.run_tests
```

### 성능 지표
- **성공률**: 태스크 완료 비율
- **효율성**: 단계 수 대비 성공률
- **안정성**: 에러 발생 빈도
- **학습 속도**: 개선 곡선

## 참고 자료

### 논문 및 연구
- [Agent Q: Advanced Reasoning and Learning](https://arxiv.org/abs/2408.07199)

### 추가 학습 자료
- Playwright 공식 문서
- OpenAI API 가이드
- MCTS 알고리즘 이론
- 강화학습 기초

---

이 가이드를 통해 AgentQ 프로젝트의 전체적인 구조를 이해하고, 단계별로 구현해 나갈 수 있습니다. 각 단계마다 충분한 테스트와 검증을 거쳐 안정적인 시스템을 구축하시기 바랍니다.
