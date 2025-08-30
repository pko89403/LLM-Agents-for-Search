이 문서는 고급 파이썬 개발자를 대상으로 하며, Ollama 통합, 상태 모델 설계, LangGraph의 노드 구성과 흐름 설계, 자가-비판과 검색 루프 구현 등 핵심 설계 개념을 중점적으로 설명할 예정입니다.

개요 (Overview)

LangGraph와 Ollama 로컬 LLM을 활용하여 Agent Q 논문에서 제시된 LLM 기반 웹 에이전트 구조를 구현하는 방법을 다룹니다. Agent Q는 복잡한 웹 상호작용 환경에서 계획(Plan), 추론(Thought), 행동(Action), 설명(Explanation), **비판(Critique)**의 순환 루프를 통해 문제를 해결하는 프레임워크입니다 ￼ ￼. 본 튜토리얼에서는 모델 학습이나 DPO 알고리즘과 같은 세부 사항은 제외하고, 에이전트의 실행 구조에 초점을 맞춥니다. 우리가 만들 에이전트는 LangGraph를 사용해 이러한 단계별 루프를 구현하며, Ollama를 통해 구동되는 로컬 LLM (예: Mistral 7B 또는 LLaMA2 등)을 OpenAI ChatGPT 호환 인터페이스로 연동하여 동작합니다.

LangGraph 소개: LangGraph는 LangChain 팀에서 개발한 상태 지향 에이전트 오케스트레이션 프레임워크로, 루프와 분기를 포함한 복잡한 워크플로우를 손쉽게 구성할 수 있는 것이 장점입니다 ￼. 일반적인 비순환 체인과 달리 LangGraph는 노드 간 **사이클(Loop)**을 지원하여 에이전트의 반복적인 사고-행동 과정을 자연스럽게 모델링할 수 있습니다 ￼. 또한 상태(state) 관리와 컨텍스트 유지, 휴먼-인-더-루프(HITL) 제어 등 강력한 기능들을 제공하여 복잡한 작업에서도 신뢰성과 제어가능성을 확보합니다.

Ollama 소개: Ollama는 오픈 소스 로컬 LLM 서버로, PC나 서버 상에서 대형 언어 모델을 쉽게 구동할 수 있게 해줍니다 ￼. Ollama는 OpenAI Chat Completions API와 호환되는 인터페이스를 제공하므로, LangChain의 ChatOpenAI 클래스 등을 이용해 로컬 모델을 마치 OpenAI의 ChatGPT 모델처럼 호출할 수 있습니다 ￼. 이를 통해 인터넷 접속이 어려운 환경에서도 Mistral이나 LLaMA2 같은 강력한 언어 모델을 활용한 에이전트를 구축할 수 있습니다.

이 튜토리얼에서는 먼저 Agent Q의 Plan→Thought→Action→Explanation→Critique 구조와 각 구성 요소의 역할을 살펴본 후, LangGraph에서 상태와 노드를 정의하여 이러한 구조를 모델링하는 방법을 설명합니다. 이어서 Ollama 기반의 로컬 LLM을 LangGraph 에이전트에 통합하는 방법을 다루고, Node와 Edge를 활용하여 에이전트의 실행 흐름(플로우)을 구성하는 예제를 구현해봅니다. Critique 단계에서 조건부 분기를 사용하여 루프를 제어하는 방법과, 실제로 간단한 웹 환경(또는 시뮬레이션 환경)에서 에이전트를 실행하는 예시를 제공합니다. 마지막으로, 구현한 에이전트의 확장 및 개선 아이디어를 제안하며 튜토리얼을 마무리합니다.

Agent Q 아키텍처 이해: Plan → Thought → Action → Explanation → Critique 루프

Agent Q의 에이전트는 한 번의 질문/과제 해결을 위해 위 다섯 단계로 구성된 내부 루프를 거칩니다 ￼. 각 단계의 의미는 다음과 같습니다:
	•	Plan (계획 수립): 에이전트가 주어진 사용자 목표를 달성하기 위한 상위 단계의 계획을 수립합니다. 예를 들어 “웹에서 레스토랑 예약하기”라는 목표가 주어지면, Plan 단계에서는 “1) 예약 페이지 열기, 2) 날짜와 인원 선택, 3) 예약 완료” 와 같은 단계별 계획을 세울 수 있습니다. 실제 Agent Q 논문에서도 에이전트 출력 형식을 step-by-step 계획, 추론(Thought), 명령(Action), 상태 코드로 구성했다고 합니다 ￼. 이처럼 Plan은 전체 문제 해결의 청사진을 제공하며, 첫 번째 스텝에서 한 번 생성됩니다 (이후 반복 루프에서는 일반적으로 갱신되지 않음 ￼).
	•	Thought (추론 단계): 현재 상황에서 다음 행동을 결정하기 위한 에이전트의 내부 사고 단계입니다. Plan이 거시적 계획이라면, Thought는 그 순간에 어떤 행동을 취할지에 대한 구체적인 Chain-of-Thought 추론에 해당합니다. 예를 들어 “현재 검색 결과 페이지에 있으니, 다음으로 특정 링크를 클릭해야겠다” 와 같은 식으로, 에이전트는 자신이 세운 계획과 현재까지 얻은 정보를 바탕으로 다음 명령을 결정합니다. 이러한 추론 내용은 에이전트의 내부 메모리에 남아 이후 의사결정에 컨텍스트로 활용되지만, 직접 환경에 영향을 주지는 않습니다 ￼.
	•	Action (행동 실행): Thought 단계에서 결정한 행동을 실제 환경에 수행합니다. 웹 에이전트의 예에서는 웹 브라우저 상의 액션이 해당됩니다. 예컨대 “검색어 입력 후 엔터”, “예약 버튼 클릭”, “다음 페이지 스크롤” 등의 구체적인 명령이 Action 단계에서 실행됩니다. 이때 Action의 결과로 환경 상태(예: 웹 페이지 내용)가 변화하거나 새로운 **관찰(Observation)**이 주어집니다. Agent Q에서는 명령을 CLICK[element], TYPE[text] 등의 포맷으로 구조화하여 출력하는데, 이 부분이 실제 환경과 상호작용하는 환경 액션에 해당합니다 ￼.
	•	Explanation (설명/해설): 환경에 행동을 수행한 직후, 에이전트는 그 행동의 결과를 해석하거나 자신의 상태를 서술하는 단계를 거칩니다 ￼. 이 단계는 일종의 내부 독백 역할을 하여, 방금 취한 행동으로 얻은 정보를 요약하거나 그 의미를 정리합니다. 예를 들어 “검색 결과 여러 개가 나왔는데, 그 중 첫 번째 링크에 답이 있을 것 같다”, “예약 날짜를 5월 22일로 선택함” 등의 설명을 생성합니다. 이 Explanation 역시 Thought와 마찬가지로 내부 상태로서 기록되어 이후 단계의 컨텍스트로 활용됩니다 ￼. 설명 단계 자체가 환경을 변화시키지는 않지만, 에이전트의 의도와 상황을 명확히 표현함으로써 다음 추론에 도움을 줍니다.
	•	Critique (자체 평가): 루프의 마지막 단계로, 현재까지의 진행 상황을 평가하여 목표 달성 여부를 판단하고, 필요하면 계획을 조정하거나 추가 행동을 결정합니다. Agent Q 프레임워크의 핵심 특징 중 하나가 이 Self-Critique 메커니즘으로, 에이전트가 자신의 실패로부터 학습하고 개선할 수 있게 해준다는 점입니다 ￼ ￼. 예컨데, *“방금 얻은 답이 정확한가? 혹은 추가 정보가 필요한가?”*를 에이전트 스스로 검토합니다. 이 평가를 바탕으로 루프를 반복할지 종료할지 결정하게 됩니다. Agent Q 논문에서는 이러한 자체 비판적 단계를 활용하여 잘못된 시도에서도 피드백을 얻고 다음 행동의 전략을 개선했다고 설명합니다 ￼. 특히 복잡한 환경에서는 한 번의 실수로도 전체 작업이 실패할 수 있기 때문에, 실패한 시도를 분석하고 같은 오류를 반복하지 않도록 하는 자체 평가 단계는 성능 향상에 매우 중요합니다 ￼.

이러한 다섯 가지 구성 요소는 Plan → Thought → Action → Explanation → Critique 순서로 실행되며, Critique 단계의 판단에 따라 루프를 **반복(Loop)**하게 됩니다. Critique가 “목표를 달성하지 못했다”거나 “추가 행동이 필요하다”고 평가하면 Thought 단계로 되돌아가 다음 액션을 결정하고, 그렇지 않고 “문제가 해결되었다”고 평가하면 루프를 종료하고 에이전트가 최종 답변이나 작업 완료를 보고하게 됩니다 ￼ ￼. 이러한 구조는 인간이 문제를 해결할 때 계획을 세우고→추론하고→행동에 옮긴 뒤→결과를 보고→잘 되었는지 반성하여 다음 행동을 결정하는 과정과 유사합니다. 연구에 따르면 LLM에게 이러한 자기 성찰(self-reflection) 과정을 부여하면 한 번에 정답을 내지 못하는 경우에도 해결률이 크게 향상된다고 합니다 ￼. Agent Q 역시 이러한 루프와 자체 평가를 통해 복잡한 웹 탐색 문제를 높은 성공률로 풀어냈음이 보고되었습니다 ￼.

LangGraph로 상태(State) 및 노드(Node) 모델링

LangGraph에서는 위와 같은 에이전트 제어 흐름을 그래프(Graph) 형태로 정의합니다. LangGraph의 핵심 개념은 상태(State), 노드(Node), 엣지(Edge) 세 가지로 요약할 수 있습니다 ￼:
	•	상태 (State): 에이전트 실행 중 공유되는 전역 상태로, 현재까지의 모든 정보를 포함합니다. State는 Python의 TypedDict나 데이터클래스(dataclass) 등으로 스키마를 정의하며, 에이전트가 다루는 데이터(예: 사용자 입력, 중간 결과, 메모리 등)를 구조화합니다 ￼. State 객체는 각 노드 간에 전달되며, 노드 실행 결과로 state가 갱신됩니다. (LangGraph는 각 state 필드별로 리듀서 함수를 지정하여, state 업데이트 시 덮어쓸지 합칠지 등의 동작을 정의할 수도 있습니다.)
	•	노드 (Node): 에이전트 동작의 개별 단계를 나타내며, 하나의 노드는 하나의 파이썬 함수로 구현됩니다 ￼. Node 함수는 입력으로 현재 state를 받아서 어떤 처리를 수행한 뒤(예: LLM 호출, 계산, 도구 사용 등) 갱신된 state 조각을 반환합니다 ￼. LangGraph에서 노드는 에이전트의 논리적인 작업 단위로 생각할 수 있습니다. 예를 들어 Plan, Thought, Action 각각을 하나의 Node로 구현하게 되며, Node 내부에서 LLM을 호출하거나 환경과 상호작용하는 코드를 작성합니다. Node 함수의 반환값(dict 형태)은 state에 merge되어 다음 상태로 반영됩니다.
	•	엣지 (Edge): 어떤 노드가 실행된 후 다음에 어떤 노드로 이동할지를 결정하는 규칙입니다 ￼. LangGraph에서는 각 엣지도 함수로 정의되는데, 가장 간단한 경우 고정된 다음 노드를 지정하는 직접 엣지(direct edge)로 사용할 수 있고, 보다 복잡한 경우 현재 state를 보고 분기하는 조건부 엣지(conditional edge)로 사용할 수도 있습니다 ￼ ￼. 예를 들어, Thought → Action으로 항상 넘어가는 고정 경로는 일반 엣지로 표현하고, Critique 노드 이후 state[“done”] 여부에 따라 End로 갈지 Thought로 돌아갈지를 결정하는 부분은 조건부 엣지로 표현할 수 있습니다. Node가 여러 개의 출력 경로를 가질 수도 있고 (복수 Edge 발송 시 병렬 실행 가능), 어떤 경로도 선택되지 않아 그래프 실행을 종료할 수도 있습니다 ￼.

요약하면: **“Node가 실제 작업을 수행하고, Edge가 다음에 실행할 노드를 결정한다”**고 할 수 있습니다 ￼. 이 구조를 활용하면 Plan, Thought, Action, Explanation, Critique 단계를 각각 Node로 만들고, **START → Plan → Thought → Action → Explanation → Critique → (반복 또는 종료)**의 그래프 흐름을 LangGraph 상에서 구현할 수 있습니다. Node와 Edge 모두 일반 함수이므로, LLM 호출 등의 복잡한 논리도 Node 내부에 자유롭게 넣을 수 있고, Edge에서도 파이썬 코드를 통해 분기 로직을 세밀하게 제어할 수 있습니다 ￼. LangGraph는 이러한 그래프 정의를 마친 후 graph = builder.compile() 과 같이 컴파일 단계を 거쳐 실행 가능한 Graph 객체로 만들며, graph.invoke(initial_state) 나 graph.stream(initial_state) 형태로 에이전트를 호출할 수 있습니다 ￼ ￼.

State 설계의 예시로, 우리 에이전트의 state에는 다음과 같은 필드가 포함될 것입니다:

```python
from typing_extensions import TypedDict

class AgentState(TypedDict):
    user_input: str        # 사용자 질문/명령
    plan: str              # 계획 (Plan 단계 결과)
    thought: str           # 추론 내용 (Thought 단계 결과)
    action: str            # 실행할 액션 명령 (예: "SEARCH: 프랑스 수도")
    observation: str       # 환경으로부터 얻은 관찰 결과 (Action 수행 후)
    explanation: str       # 설명/내부 해설 (Explanation 단계 결과)
    done: bool             # 완료 여부 (Critique 판단)
```

위와 같이 state 스키마를 정의해 두면, 각 Node 함수는 AgentState를 입력으로 받고 필요한 키를 업데이트하여 반환합니다. 예를 들어 Plan 노드는 plan 필드를 채우고, Thought 노드는 thought와 action을 결정하여 반환하며, Action 노드는 실제 환경 액션을 실행한 뒤 observation을 반환하는 식입니다. 마지막 Critique 노드는 done 플래그를 True/False로 설정해주어 Edge 분기에 활용합니다.

LangGraph에서는 이러한 그래프 정의를 보다 손쉽게 하기 위해 StateGraph라는 빌더 클래스를 제공합니다 ￼ ￼. graph_builder = StateGraph(AgentState) 처럼 초기화한 뒤, graph_builder.add_node("노드명", 노드함수) 형태로 노드를 추가하고, add_edge나 add_conditional_edges 메서드로 노드 간 연결을 정의합니다 ￼ ￼. 모든 노드와 엣지를 정의한 뒤 graph = graph_builder.compile()로 컴파일하여 그래프를 완성합니다 ￼. 다음 섹션에서는 이렇게 LangGraph에 노드와 엣지를 추가하면서 Agent Q의 루프를 구현해보겠습니다.

Ollama 기반 로컬 LLM 연동

에이전트의 각 단계에서 LLM 호출이 필요하므로, 로컬 LLM을 LangGraph 노드 함수 내부에서 사용할 수 있도록 세팅해야 합니다. 앞서 언급했듯 Ollama는 OpenAI Chat API와 호환되도록 설계되어 있어, LangChain의 ChatOpenAI 모델을 이용해 손쉽게 로컬 모델을 호출할 수 있습니다 ￼. 먼저, 시스템에 Ollama를 설치하고 원하는 모델을 다운로드/로드해야 합니다. 예를 들어 Mistral 7B 모델을 사용할 경우, 터미널에서 다음을 실행합니다:

```bash
# Ollama 설치 후, llama3.2 3B 모델 다운로드
ollama pull llama3.2:3b

# Ollama LLM 서버 실행 (기본 포트 11434)
ollama serve
```
Ollama가 실행되면 http://localhost:11434에서 OpenAI 호환 REST API가 열립니다 ￼. LangChain에서는 다음과 같이 로컬 LLM을 초기화할 수 있습니다:
```python
from langchain.chat_models import ChatOpenAI

# 로컬 LLM 초기화: 모델 이름과 로컬 Ollama 엔드포인트 지정
llm = ChatOpenAI(
    model="mistral-7b",                # Ollama에 로드된 모델 이름
    base_url="http://localhost:11434/v1",  # Ollama OpenAI 호환 API URL
    api_key="ollama",                 # API 키 자리에는 임의 문자열 사용
    temperature=0
)
```
위 코드에서 base_url에 Ollama의 주소를 지정하고, api_key에는 "ollama" 등의 더미 값을 넣었습니다. (Ollama 기본 설정에서는 인증이 필요 없으므로 임의의 문자열을 써도 되며, "sk-..." 형태처럼 OpenAI 키 포맷을 맞춰주기도 합니다 ￼.) 이제 llm 객체는 OpenAI의 ChatGPT 모델을 다루듯이 llm.predict(...) 또는 llm.invoke([...]) 등을 통해 프롬프트를 던지고 응답을 받을 수 있습니다. LangChain ChatOpenAI를 사용하므로 시스템 메시지나 함수 호출 등의 고급 기능도 동일하게 쓸 수 있습니다.

참고로, LangChain에서는 langchain_community 패키지의 ChatOllama 래퍼도 제공하지만, 위처럼 ChatOpenAI에 base_url을 설정하는 방식이 간편합니다. Ollama 연동이 완료되었으니, 다음으로 LangGraph 노드들에 이 LLM을 활용하는 로직을 넣어보겠습니다.

LangGraph 에이전트 흐름 구성: 노드 추가와 조건 분기

이제 Agent Q 구조의 각 단계를 LangGraph 그래프로 구현해보겠습니다. 앞서 정의한 AgentState를 사용하여 StateGraph를 초기화하고, Plan/Thought/Action/Explanation/Critique에 해당하는 Node를 차례로 추가하겠습니다. 각 Node 함수 내부에서 할 일을 정리하면 다음과 같습니다:
	•	Plan 노드: 사용자 입력 (state["user_input"])을 읽어, 문제 해결을 위한 계획을 세웁니다. LLM에게 현재 목표에 대한 단계별 계획을 묻는 프롬프트를 만들어 호출하고, 응답을 state["plan"]에 저장합니다.
	•	Thought 노드: 수립된 계획과 현재까지 얻은 정보(초기엔 계획만, 이후 루프에서는 observation도 있음)를 참고하여 다음 액션을 결정합니다. 간단히 구현하기 위해, Thought 노드는 내부적으로 “어떤 액션을 실행할지”를 결정하여 state["action"] 필드를 채우도록 하겠습니다. (실제 Agent Q에서는 Thought는 텍스트상 추론만 하고 Action은 별도로 결정하지만, 여기서는 편의상 Thought 단계에서 곧바로 다음 Action 커맨드까지 정해버리겠습니다.) 예시로, 사용자 질문이 정보 검색을 필요로 하면 Thought 노드에서 SEARCH: ... 형태의 검색 명령을 만들고, 그렇지 않으면 다른 액션을 만들 수 있습니다. 이 예제에서는 질문에 대한 답을 찾기 위해 웹 검색을 기본 액션으로 사용하겠습니다.
	•	Action 노드: Thought에서 결정한 액션 커맨드를 받아 실제로 환경에 수행합니다. 튜토리얼에서는 실제 웹 브라우저를 제어하지 않고, 간단히 모의 웹 검색 함수를 통해 시뮬레이션하겠습니다. state["action"]에 "SEARCH: ..." 형태의 문자열이 오면, 해당 검색어에 대한 가짜 검색 결과를 돌려주는 식입니다. (이를 위해 간단한 파이썬 조건문이나 사전 매핑을 사용할 것입니다.) 실행 결과는 state["observation"]에 기록합니다. 웹 환경의 다른 액션 (예: 클릭, 입력 등)은 여기 다루지 않지만, 원한다면 Action 노드에서 if/else로 여러 명령을 처리할 수 있습니다.
	•	Explanation 노드: Action 결과로 얻은 observation을 해석/요약하는 설명을 생성합니다. 검색 결과라면 유의미한 정보를 요약하고, 그 외 액션이라면 현재까지 진행 사항을 서술할 수 있습니다. 이 또한 LLM을 불러서 요약하도록 할 수도 있지만, 간단히 observation 문자열을 가공하여 "~이다" 형태로 문장화하겠습니다. 이 설명은 state["explanation"]에 저장하며, 필요하다면 최종 답변으로도 활용될 수 있습니다. (Agent Q에서는 Explanation이 내부 상태로만 쓰이지만, 튜토리얼에선 최종 답변 역할도 겸하게 할 예정입니다.)
	•	Critique 노드: 마지막으로, 현재 얻은 설명이나 결과를 보고 문제가 해결되었는지 판단합니다. 우리의 시뮬레이션에서는 state["observation"]에 원하는 답이나 결과 키워드가 포함되어 있는지를 기준으로 삼겠습니다. 예를 들어 검색 결과에 “찾는 정보가 없음”이라는 문구가 있으면 done을 False로, 그렇지 않으면 True로 설정합니다. 현실에서는 Critique 단계를 LLM 호출로 구현하여 “해결됐는가? 추가 계획이 필요한가?”를 물어볼 수 있고, 또는 규칙기반으로 특정 조건을 검사할 수도 있습니다. Critique 노드는 state["done"] = True/False를 반환하며, 이 값은 다음 Edge에서 루프 반복 여부를 결정하는데 사용됩니다.

이제 이러한 로직을 바탕으로 코드를 작성해보겠습니다:
```python
from langgraph.graph import StateGraph, START, END

# 1. 그래프 초기화
builder = StateGraph(AgentState)

# 2. Plan 노드 추가
def plan_node(state: AgentState) -> dict:
    user_query = state["user_input"]
    prompt = f"다음 목표에 대한 단계별 계획을 세워줘: '{user_query}'"
    plan_text = llm.predict(prompt)  # LLM 호출로 계획 수립 (예: "1. ...\n2. ...")
    return {"plan": plan_text.strip()}
builder.add_node("plan", plan_node)

# 3. Thought 노드 추가 (다음 행동 및 추론)
def thought_node(state: AgentState) -> dict:
    # 예시: 무조건 웹 검색 액션을 선택
    query = state["user_input"]
    thought_text = f"질문의 답을 찾기 위해 웹에서 '{query}'를 검색해야겠다."
    action_cmd = f"SEARCH: {query}"
    return {"thought": thought_text, "action": action_cmd}
builder.add_node("thought", thought_node)

# 4. Action 노드 추가 (환경 액션 수행 시뮬레이션)
def action_node(state: AgentState) -> dict:
    action_cmd = state["action"]
    observation = ""
    if action_cmd.startswith("SEARCH:"):
        query = action_cmd[len("SEARCH:"):].strip()
        # 매우 간단한 검색 결과 모킹: "프랑스" 키워드 여부로 결과 결정
        if "프랑스" in query.lower() or "france" in query.lower():
            observation = "파리는 프랑스의 수도입니다."  # 원하는 정보 발견
        else:
            observation = f"'{query}'에 대한 유의미한 정보를 찾지 못했다."
    else:
        observation = f"Executed action: {action_cmd}"
    return {"observation": observation}
builder.add_node("action", action_node)

# 5. Explanation 노드 추가 (결과 설명/요약)
def explanation_node(state: AgentState) -> dict:
    obs = state.get("observation", "")
    if not obs:
        explanation = "방금 수행한 행동으로 새로운 정보를 얻지 못했다."
    elif "없" in obs or "못했다" in obs:
        explanation = f"검색 결과 유의미한 정보를 얻지 못했다. 추가 조치가 필요하다."
    else:
        explanation = f"검색 결과에 따르면 {obs}"
    return {"explanation": explanation}
builder.add_node("explanation", explanation_node)

# 6. Critique 노드 추가 (완료 여부 판단)
def critique_node(state: AgentState) -> dict:
    explanation = state.get("explanation", "")
    done_flag = False
    # 설명에 원하는 답이 포함됐는지 혹은 추가 조치 필요 문구가 있는지 검사
    if explanation and "없" not in explanation and "추가 조치" not in explanation:
        done_flag = True   # 답을 얻었다고 가정
    return {"done": done_flag}
builder.add_node("critique", critique_node)

# 7. 노드 간 엣지 연결 (그래프 흐름 정의)
builder.add_edge(START, "plan")          # 시작 시 Plan 실행
builder.add_edge("plan", "thought")      # Plan 완료 후 Thought로
builder.add_edge("thought", "action")    # Thought 후 Action으로
builder.add_edge("action", "explanation")# Action 후 Explanation으로
builder.add_edge("explanation", "critique") # Explanation 후 Critique로

# Critique 이후 조건부 분기: done=True이면 종료, False이면 Thought로 루프
def check_done(state: AgentState):
    return state["done"]
builder.add_conditional_edges("critique", check_done,
                              {False: "thought", True: END})

# 8. 그래프 컴파일
graph = builder.compile()
```

위 코드에서는 Plan→Thought→Action→Explanation→Critique 순으로 노드를 추가하고, 마지막에 critique 노드에서 조건부 엣지를 설정한 모습에 주목하십시오. check_done 함수는 현재 state의 "done" 값을 반환하며 ￼, add_conditional_edges("critique", check_done, {False: "thought", True: END})를 통해 done이 False일 때 thought 노드로 돌아가고, True일 때 END 노드로 연결되도록 했습니다 ￼ ￼. 여기서 END는 LangGraph에서 그래프 종료를 표시하는 특수 노드 상수로, 해당 노드로 연결하면 더 이상 수행할 노드가 없으므로 실행이 종료됩니다 ￼ ￼. 이로써 Critique 결과에 따라 루프 반복/종료가 결정되는 흐름을 완성했습니다.

지금까지 정의한 그래프를 도식화하면 다음과 같습니다:
	•	START → Plan → Thought → Action → Explanation → Critique
↳ (if done=False) Thought (다음 루프 반복)…
↳ (if done=True) END (종료)

즉, Plan 노드는 처음에 한 번 실행되고, Critique 노드가 done=False를 반환하면 Thought 노드로 되돌아가 새로운 액션을 결정합니다. 이 과정에서 이전까지의 state (계획, 직전까지의 설명 등)이 모두 보존되므로, 루프를 거칠수록 state에는 점점 더 풍부한 문맥이 쌓입니다. LangGraph는 state를 지속적으로 축적하면서 각 노드 함수가 필요한 정보에 접근할 수 있게 해줍니다 (예를 들어 Thought 노드에서 state["explanation"]을 참고하여 새로운 결정을 내릴 수도 있습니다). 또한 LangGraph는 기본적으로 각 노드 실행 후 state를 자동 저장(checkpoint)하거나, 실행 과정을 스트리밍 출력하는 기능도 제공하므로, 긴 루프를 돌리는 에이전트의 중간 진행 상황을 추적하기 용이합니다 ￼ ￼.

코드 마지막에 graph = builder.compile()로 그래프를 컴파일했고, 이제 이 graph 객체를 실제로 호출하여 에이전트를 동작시킬 수 있습니다. graph.invoke({...})에 초기 state(최소한 user_input 포함)를 넘기면, 모든 노드들이 순서대로 실행되고 최종 state 결과를 반환합니다. 또는 graph.stream({...})을 사용하면 실행 슈퍼스텝 단위로 스트리밍 이벤트를 얻어볼 수도 있습니다 ￼.

Critique를 활용한 루프 반복 및 중단 조건 설계

위에서 구현한 Critique 노드는 에이전트 루프의 조절자 역할을 합니다. LangGraph의 조건부 엣지를 통해 Critique 노드의 결과에 따라 다음 단계가 달라지게 했는데, 이를 일반화하면 **“종료 조건을 판단하는 노드”**를 두어 에이전트의 반복 실행을 제어할 수 있다는 뜻입니다. Critique 단계의 설계 포인트 및 변형 가능성을 정리하면 다음과 같습니다:
	•	종료 조건 설정: 우리의 예제에서는 매우 단순히 state["done"] 불리언 값으로 종료 여부를 표현했습니다. 복잡한 시나리오에서는 이 값을 결정하기 위해 여러 정보를 종합해야 할 수도 있습니다. 예를 들어 다단계 임무에서 각 하위 목표의 완료 여부를 추적하거나, 혹은 최대 반복 횟수를 정해놓고 그 이상 루프를 돌지 않도록 할 수 있습니다. 이러한 로직을 Critique 노드 (또는 Edge의 routing 함수)에서 구현하면 됩니다. LangGraph에서는 Edge의 분기 함수 내에서도 state를 얼마든지 검사할 수 있으므로, add_conditional_edges의 routing 함수에서 직접 복잡한 조건들을 체크해 True/False나 특정 노드 이름을 리턴하도록 해도 됩니다 ￼.
	•	자체 평가 LLM 활용: Critique를 사람의 검토자처럼 활용하려면, LLM에게 *“지금까지 결과로 보아 목표를 달성했는지, 안 됐다면 무엇이 문제인지”*를 물어보는 프롬프트를 구성할 수도 있습니다. 예를 들어 Explanation까지의 히스토리를 모두 맥락으로 주고 *“위 대화에서 오류나 부족한 부분이 있으면 지적하고, 추가 행동이 필요하면 ‘NO’를 답하라”*는 식으로 물어볼 수 있습니다. LLM이 “해결됨/안됨” 판단과 함께 조언까지 주도록 할 수도 있겠지요. 이러한 Reflective Critique 기법은 많은 연구에서 LLM의 성능 향상에 도움이 되는 것으로 보고되었습니다 ￼. 다만 LLM의 응답을 신뢰할 수 있어야 하므로, 필요하다면 Critique 결과를 이중 체크하거나, 아주 보수적으로 설계할 수 있습니다.
	•	루프 탈출 조건: Critique 노드가 항상 False를 반환하면 무한 루프에 빠질 것입니다. 이를 방지하기 위해 보통 한 번의 사용자 요청에 대해 최대 loop 반복 횟수를 정해놓습니다. LangGraph에서는 이러한 제약도 쉽게 추가할 수 있습니다. 하나의 방법은 state에 loop_count 필드를 두고, Thought 노드 실행 시마다 +1 증가시키게 한 후 Critique에서 loop_count가 특정 값 이상이면 강제로 done=True 설정을 하는 것입니다. 또 다른 방법으로 LangGraph RuntimeContext에서 제공하는 Recursion Limit 등을 활용하여 그래프 자체에 최대 반복 깊이를 설정할 수도 있습니다 ￼. 우리 예제에서는 loop가 길어질 일이 없으므로 생략했지만, 실전에서는 무한 루프 방지 장치를 넣는 것이 중요합니다.
	•	Human-in-the-loop 및 종료 시그널: Agent Q처럼 완전 자동화된 에이전트가 아니라, 필요 시 사람에게 컨펌을 받는 구조를 원한다면 Critique 단계에서 인간 개입을 요청할 수도 있습니다. LangGraph는 특정 노드에서 실행을 일시 정지하고 외부 입력을 기다리는 기능도 지원합니다 ￼. 예컨대 Critique 노드에서 *“사용자에게 중간 결과를 검토받을까요?”*라고 판단하면 state["pause"] = True 같은 플래그를 세팅해두고, 애플리케이션 레벨에서 이를 감지해 사람에게 제어를 넘기는 식으로 확장 가능합니다.

정리하면, Critique 단계는 에이전트의 자율성 수준과 종료 조건을 결정짓는 중요한 부분입니다. Agent Q 연구에서는 이 Critique를 한층 발전시켜, 에이전트가 성공 trajactory 뿐만 아니라 실패 trajactory도 학습하도록 했습니다 ￼. 즉, Critique를 통해 “어디서 잘못됐는지”까지 자기 성찰하게 하고, 이러한 경험을 DPO(off-policy RL)로 최종 정책에 반영함으로써 실패 사례로부터도 배우는 고도화된 전략을 취했습니다 ￼ ￼. 비록 본 튜토리얼에서는 학습 부분은 다루지 않았지만, 루프 내 자체 평가는 에이전트의 성능을 좌우할 수 있는 강력한 도구임을 염두에 두세요.

실행 예시: 시뮬레이션으로 보는 에이전트 동작

구현한 LangGraph 에이전트를 실제로 실행하여, Plan→…→Critique 루프가 도는 과정을 살펴보겠습니다. 예제로 “프랑스의 수도는 어디인가요?” 라는 사용자 질문을 주어보겠습니다:
	•	User 질문: “프랑스의 수도는 어디인가요?”
	•	Agent Plan: (LLM 생성) “1. 웹에서 프랑스의 수도를 검색한다. 2. 검색 결과에서 수도 정보를 찾는다. 3. 찾은 정보를 사용자에게 전달한다.” ￼
(에이전트가 단계별 계획을 세움)
	•	Agent Thought: “프랑스의 수도를 알아내기 위해 먼저 웹 검색을 해야겠다.”
(계획에 따라 다음 행동으로 웹 검색을 선택)
	•	Agent Action: 검색 엔진에서 "프랑스 수도"를 검색 🔍
(시스템이 검색을 수행함 - 모의 환경)
	•	Agent Observation: “파리는 프랑스의 수도입니다.”
(검색 결과로 해당 정보를 얻음)
	•	Agent Explanation: “검색 결과에 따르면 프랑스의 수도는 파리이다.”
(방금 얻은 정보를 내재화하여 요약함)
	•	Agent Critique: ”(평가) 필요한 답을 얻었으므로 이제 종료하면 되겠다.” ￼
(자체 평가 결과, 목표 달성 완료로 판단 → 루프 종료)
	•	Agent 최종 답변: “프랑스의 수도는 파리입니다.”

위 시나리오에서 볼 수 있듯, 에이전트는 Plan 단계에서 전체 전략을 세우고, Thought→Action→Explanation을 거쳐 필요한 정보를 찾아냈습니다. Critique 단계에서는 Explanation에 원하는 답이 포함되어 있으므로 추가 반복 없이 종료를 결정합니다. 만약 첫 번째 검색에서 정보를 찾지 못했다면 Critique가 done=False를 내리고 다시 Thought 단계로 돌아가 다른 검색어로 재시도하거나, Plan을 재고하는 흐름이 되었을 것입니다. (예를 들어 “프랑스의 수도를 찾는 첫 번째 시도는 실패했다. 다른 검색어나 방법을 시도해보자.” 라는 Thought를 생성하고, Action으로 두 번째 검색을 수행하는 식으로 진행합니다.) 이렇듯 에이전트는 자체 루프를 통해 점진적으로 문제에 접근하며, 실패하더라도 복구를 시도할 수 있습니다 ￼.

우리의 예제는 단순 정보 질의였지만, Agent Q가 다룬 웹 예약 시나리오에서도 동일한 원리가 적용됩니다. 예컨대 “오픈테이블에서 Cecconi’s 레스토랑 4명 예약” 과제가 들어오면, Plan 단계에서 필요한 큰 단계들을 나열합니다 (테이블 찾기 → 시간 선택 → 예약 완료 등). Thought/Action 단계들을 통해 웹 폼에 입력하고 결과 페이지를 확인하며 진행하지요. 그리고 Critique에서 예약이 성공적으로 됐는지 확인하여 완료 여부를 판단합니다. Agent Q의 실험에 따르면 이러한 과정을 거친 에이전트는 사람과 유사한 수준의 성과를 내었고, 특히 Critique를 통해 오류를 수정하면서 최종적으로 성공률을 크게 끌어올렸다고 합니다 ￼.

확장 및 개선 방향 제안

이번 튜토리얼에서는 LangGraph와 로컬 LLM을 활용해 Agent Q의 핵심 아이디어인 계획-행동-자기성찰 루프를 구현해보았습니다. 실제 응용이나 연구에 적용하기 위해 고려할 수 있는 확장 방향을 몇 가지 제안합니다:
	•	사실적인 웹 액션 통합: 튜토리얼에서는 검색을 단순 모킹했지만, 실제 웹 브라우저 조작은 Selenium이나 Playwright 같은 도구, 혹은 LangChain의 Browser toolkit 등을 사용할 수 있습니다. LangGraph에서 도구 통합은 간단한데, langchain_community.tools의 도구 클래스를 활용하거나 Python 함수를 툴로 묶어 llm.bind_tools()로 LLM에 연결하면 됩니다 ￼ ￼. 예를 들어 Tavily API를 통한 실제 웹 검색 툴을 Action 노드에서 호출하게 하거나, Browser automation 함수를 만들어 쓸 수 있습니다. 이러한 실제 액션 통합으로 에이전트를 현실 웹 환경에서 동작시킬 수 있습니다.
	•	복잡한 플래닝 기법 접목: Agent Q 논문에서는 MCTS(Monte Carlo Tree Search) 알고리즘을 활용해 에이전트의 장기 플랜 탐색을 향상시켰습니다 ￼ ￼. 우리 구현에도 이를 응용하려면, Thought 단계에서 한 번에 1개의 액션이 아니라 복수의 액션 후보를 생성하고, Critique 단계에서 각 후보에 대해 시뮬레이션하거나 평가 점수를 매겨 가장 유망한 경로를 선택하는 식으로 확장할 수 있습니다. LangGraph는 병렬 실행과 브랜치 관리가 가능하므로 (하나의 노드에서 여러 엣지로 메시지를 broadcast 가능 ￼), MCTS 같은 트리 탐색을 구현하는 것도 이론적으로 가능할 것입니다. 다만 이는 구현 복잡도가 높으므로 별도 모듈로 분리하는 편이 나을 수 있습니다.
	•	메모리 및 상태 관리 강화: 복잡한 웹 상호작용일수록 에이전트가 다뤄야 할 컨텍스트 양이 방대해집니다. LangGraph는 state에 메시지 기록이나 요약 메모리 등을 자유롭게 포함할 수 있으므로, 필요한 경우 Memory 모듈과 연계해 history를 요약하면서 가져가는 것도 고려해야 합니다. 예를 들어 Explanation 내용을 계속 쌓으면 컨텍스트가 커질 테니, 이전 Explanation들은 요약해서 저장하거나, 중요한 정보(예: 예약 상세 정보)는 별도 키로 보존하여 필요 시 참조하도록 state를 구성할 수 있습니다.
	•	모델 교체 및 다중 LLM 활용: 이번 구현에서는 하나의 LLM(Mistral 7B)을 모든 단계에 사용했지만, 경우에 따라 역할별로 다른 모델을 쓸 수도 있습니다. 예를 들어 Plan/Thought 같이 고차원적 추론은 큰 모델(GPT-4 등)에 맡기고, Action 단계의 세부 파싱은 작은 모델이나 규칙기반 코드로 처리하는 식입니다. LangGraph는 멀티 에이전트/모델 구조도 지원하므로, 노드별로 다른 LLM 객체를 사용하게끔 구성할 수도 있습니다. 또한 속도를 높이기 위해 Plan 단계만 한 번 클라우드 API를 쓰고 나머지는 로컬 모델로 한다든지 혼합 운영도 가능할 것입니다.
	•	인간 피드백 및 학습 적용: Agent Q의 궁극적인 목표는 자율적으로 경험을 쌓아 학습하는 에이전트입니다. 이를 위해 인트라넷 환경에서 자체적으로 DPO (Direct Preference Optimization) 파이프라인으로 개선을 거듭했는데 ￼, 우리 에이전트에도 사용자의 피드백이나 성공/실패 로그를 축적하여 모델을 fine-tuning하는 피드백 루프를 붙일 수 있습니다. LangGraph는 실행 중간에 사용자 확인 단계를 넣는 것도 가능하므로, Critique 단계에서 사용자의 정정(feedback)을 받아 state에 반영하고 다음 루프에서 이를 고려하게 할 수도 있습니다.

마지막으로, 에이전트 성능을 높이는 실용적인 팁을 덧붙이자면: 프롬프트 엔지니어링과 테스트를 반복하세요. Plan이나 Critique 단계의 프롬프트를 어떻게 주느냐에 따라 에이전트의 행동이 크게 달라질 수 있습니다. 예를 들어 Critique 프롬프트에 *“틀렸으면 NO라고 답해”*라고 하면 매우 보수적으로 굴 것이고, *“대부분 맞았으면 YES”*라고 하면 쉽게 끝낼 것입니다. 이러한 미세 조정은 도메인 지식에 따라 달라지므로, 에이전트를 충분히 시험해보고 루프 거동을 관찰하면서 최적의 설정을 찾아야 합니다. LangGraph의 스트리밍 실행 기능으로 에이전트의 단계별 동작과 token 사용 등을 시각화해가며 디버깅하면 많은 도움이 됩니다.

以上, 고급 파이썬 개발자를 위한 LangGraph와 Ollama 기반 Agent Q 구현 튜토리얼을 마칩니다. 이 튜토리얼을 통해 복잡한 LLM 에이전트의 구조를 이해하고, 로컬 환경에서 실험해볼 수 있는 기반을 얻으셨기를 바랍니다. Agent Q와 같은 자율 에이전트 연구는 여전히 발전 중인 분야이며, 오늘 다룬 개념들을 응용해 더욱 강력하고 똑똑한 에이전트를 직접 만들어보시길 권장합니다. 성공과 실패를 모두 배우며 성장하는 에이전트를 만드는 도전, 이제 여러분의 손에 달렸습니다! ￼ ￼

참고 문헌: Agent Q 논문 ￼ ￼, LangChain LangGraph 공식 문서 ￼ ￼, NVIDIA 기술 블로그, 기타 LLM 에이전트 관련 연구 및 Medium 글 ￼ ￼ 등. (각 출처는 본문에 인용으로 표시)
