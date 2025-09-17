---

## **CodeAct 에이전트 LangGraph 재구현 가이드**

### 1. 개요
이 문서는 "Executable Code Actions Elicit Better LLM Agents" 논문에서 제안된 **CodeAct** 메소드의 핵심 기능을 **LangGraph**를 사용하여 재구현하는 방법을 안내합니다.

논문에 따르면, 기존 LLM 에이전트가 사용하는 JSON이나 텍스트 기반의 행동(action)은 미리 정의된 기능에만 묶여있어 유연성과 확장성에 한계가 있습니다. CodeAct는 이 문제를 해결하기 위해 **실행 가능한 Python 코드를 에이전트의 통합된 행동 표현으로 사용**할 것을 제안합니다. 에이전트가 생성한 코드는 인터프리터 환경에서 실행되며, 그 실행 결과(Observation)는 다시 에이전트에게 피드백됩니다. 이 다중 턴(multi-turn) 상호작용을 통해 에이전트는 스스로 행동을 수정하거나, 여러 도구를 조합하는 등 복잡하고 정교한 작업을 수행할 수 있게 됩니다.

본 가이드는 기존 CodeAct 프로젝트(`scripts/chat/demo.py`의 `while` 루프)의 절차적 로직을 LangGraph의 상태 기반 그래프(Stateful Graph)로 전환하여, CodeAct의 상호작용 사이클을 보다 명확하고 모듈식으로 구현하는 것을 목표로 합니다.

### 2. 핵심 아키텍처 비교

| 구분 | 기존 CodeAct 구현 방식 (`demo.py` 기반) | LangGraph 구현 방식 |
| --- | --- | --- |
| **상태 관리** | `while` 루프 내에서 대화 히스토리(list)를 직접 관리 | `StateGraph`의 `state` 객체로 대화 히스토리 및 중간 결과 관리 |
| **에이전트 (LLM)** | `openai_lm_agent.py` 등에서 모델을 직접 호출하고 결과 파싱 | 그래프의 **`Node`**로 정의. 상태를 입력받아 LLM을 호출하고 결과를 상태에 추가. |
| **코드 실행기** | `python_tool.py`가 `jupyter.py`를 통해 Jupyter 커널에서 코드 실행 | 그래프의 **`Tool Node`**로 정의. 에이전트가 생성한 코드를 실행하고 결과를 반환. |
| **흐름 제어** | `if/else` 문으로 코드 실행 여부 및 종료 결정 | **`Conditional Edge`**를 사용하여 LLM의 출력에 따라 다음 노드(코드 실행 or 종료)를 동적으로 결정. |

### 3. LangGraph 구현 단계

LangGraph를 사용한 구현은 크게 **상태 정의, 노드 정의, 엣지 연결**의 3단계로 나뉩니다.

#### **1단계: Graph 상태(State) 정의**
그래프의 각 노드를 거치며 전달되고 업데이트될 데이터 구조를 정의합니다. CodeAct의 상호작용을 위해서는 LLM에 전달될 메시지 목록이 필수적입니다.

```python
from typing import List, TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
import operator

class AgentState(TypedDict):
    """
    그래프 전체에서 공유될 상태 객체입니다.
    - messages: LLM과의 대화 기록 전체를 담습니다.
    """
    # operator.add는 새로운 메시지가 기존 리스트에 '추가'되도록 하여 상태를 업데이트합니다.
    messages: Annotated[List[BaseMessage], operator.add]
```

#### **2단계: Agent 노드 구현**
사용자 입력과 이전 기록을 바탕으로 LLM을 호출하여 다음 행동(코드 또는 최종 답변)을 생성하는 노드입니다. 이 노드는 기존 `mint/agents/openai_lm_agent.py`의 모델 호출 로직과 `scripts/chat/demo.py`의 프롬프트 구성 방식을 통합합니다.

```python
from langchain_openai import ChatOpenAI

# CodeActAgent 모델을 로드합니다.
# vLLM 또는 llama.cpp로 실행된 OpenAI 호환 API 엔드포인트를 사용합니다.
# (사전에 ./scripts/chat/start_vllm.sh 등을 통해 모델 서버가 실행되어 있어야 합니다.)
model = ChatOpenAI(
    openai_api_base="http://localhost:8080/v1",
    openai_api_key="EMPTY",
    model_name="xingyaoww/CodeActAgent-Mistral-7b-v0.1",
    temperature=0.7,
    stop=["

## Observation"], # 에이전트가 Observation을 기다리도록 stop token 설정
)

def agent_node(state: AgentState):
    """
    현재까지의 대화 상태(messages)를 기반으로 LLM을 호출하여 다음 메시지를 생성합니다.
    기존 CodeAct의 프롬프트 구조와 같이 System, User, Assistant, Observation 턴을 모두 포함하여 호출합니다.
    """
    print("Calling Agent Node...")
    # LLM이 생성한 새로운 AI 메시지를 상태에 추가하여 반환합니다.
    response = model.invoke(state["messages"])
    return {"messages": [response]}
```
**구현 설명**: `agent_node`는 현재 `state`의 `messages` 리스트 전체를 LLM에 전달합니다. 이 리스트에는 초기 시스템 프롬프트, 사용자의 질문, 이전 AI의 답변(코드 포함), 그리고 코드 실행 결과(Observation)가 순서대로 포함되어야 합니다. 이는 CodeAct가 다중 턴 상호작용을 통해 컨텍스트를 유지하는 핵심 방식입니다.

#### **3단계: 코드 실행 도구(Tool) 노드 구현**
Agent 노드가 생성한 Python 코드를 실제로 실행하고 결과를 반환하는 노드입니다. 이 노드는 기존 `mint/tools/python_tool.py`와 `scripts/chat/code_execution/jupyter.py`의 로직을 내장한 `CodeExecutor` 클래스를 사용하여 구현합니다.

```python
import re
import requests
import json

class CodeExecutor:
    """
    기존 CodeAct의 Jupyter Kernel Gateway 연동 방식을 클래스로 캡슐화합니다.
    - __init__: Jupyter 서버 URL을 받아 세션을 초기화합니다.
    - execute: 코드 문자열을 받아 서버에 전송하고, 실행 결과를 포맷팅하여 반환합니다.
    """
    def __init__(self, kernel_url: str):
        self.kernel_url = kernel_url
        self.session = requests.Session()

    def execute(self, code: str) -> str:
        print(f"Executing Code:
{code}")
        try:
            response = self.session.post(
                self.kernel_url,
                json={"code": code, "stop_on_error": True},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            # 기존 python_tool.py의 결과 포맷팅 로직을 참고하여 Observation 문자열 생성
            output = ""
            if result.get("stdout"):
                output += f"**stdout**:
```
{result['stdout']}
```
"
            if result.get("stderr"):
                output += f"**stderr**:
```
{result['stderr']}
```
"
            if "result" in result:
                 output += f"**Result**:
```
{result['result']}
```
"
            
            return output if output else "Code executed successfully with no output."

        except requests.exceptions.RequestException as e:
            return f"**Execution Error**: {e}"
        except Exception as e:
            return f"**An unexpected error occurred**: {e}"

# 코드 실행기 인스턴스 생성 (Jupyter 서버는 8081 포트에서 실행 중이라고 가정)
# (사전에 ./scripts/chat/code_execution/start_jupyter_server.sh 8081 실행 필요)
code_executor = CodeExecutor(kernel_url="http://localhost:8081/execute")

def code_tool_node(state: AgentState):
    """
    가장 최근의 AI 메시지에서 코드를 추출하여 실행하고,
    그 결과를 'Observation' 메시지로 변환하여 상태에 추가합니다.
    """
    print("Calling Code Executor Node...")
    last_message = state["messages"][-1]
    code_block_match = re.search(r"```python
(.*?)
```", last_message.content, re.DOTALL)

    if code_block_match:
        code_to_execute = code_block_match.group(1)
        observation_result = code_executor.execute(code_to_execute)
        
        # 실행 결과를 HumanMessage(name="observation")으로 래핑하여 다음 LLM 호출 시 컨텍스트로 제공
        return {"messages": [HumanMessage(content=f"## Observation
{observation_result}", name="observation")]}
    
    # 실행할 코드가 없는 경우, 아무것도 하지 않음
    return {}
```

#### **4단계: 조건부 엣지(Edge) 설정**
Agent 노드의 출력에 따라 다음 단계를 결정하는 라우터 함수입니다. 이 로직은 `scripts/chat/demo.py`의 `while` 루프 내 분기 처리 로직을 대체합니다.

```python
def should_continue(state: AgentState):
    """
    Agent의 마지막 응답을 보고 다음 경로를 결정합니다.
    - 응답에 코드 블록이 있으면 'execute_code'를 반환하여 code_tool_node로 라우팅합니다.
    - 코드 블록이 없으면 에이전트가 최종 답변을 한 것으로 간주하고 'end'를 반환하여 워크플로우를 종료합니다.
    """
    print("Checking for code in agent's response...")
    last_message = state["messages"][-1]
    if "```python" in last_message.content:
        return "execute_code"
    else:
        return "end"
```

#### **5단계: 그래프 빌드 및 실행**
이제 정의된 상태, 노드, 엣지를 연결하여 전체 워크플로우를 완성하고 실행합니다.

```python
from langgraph.graph import StateGraph, END

# 그래프 객체 생성
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("agent", agent_node)
workflow.add_node("code_executor", code_tool_node)

# 엣지 연결
workflow.set_entry_point("agent") # 시작점은 agent 노드

# 조건부 엣지: agent 노드 실행 후, should_continue 함수 결과에 따라 분기
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "execute_code": "code_executor", # 'execute_code'가 반환되면 code_executor 노드로
        "end": END                      # 'end'가 반환되면 종료
    }
)

# 일반 엣지: 코드 실행 후에는 항상 다시 agent 노드로 돌아가서 다음 행동을 생성
workflow.add_edge("code_executor", "agent")

# 그래프 컴파일
app = workflow.compile()

# --- 그래프 실행 ---
# 초기 시스템 프롬프트 설정 (기존 CodeAct 프롬프트 참고)
system_prompt = "You are a helpful AI assistant that can use a Python code interpreter to answer questions."
initial_user_query = "현재 디렉토리에 있는 파일 목록을 보여주고, 그 중 'README.md' 파일의 처음 3줄을 읽어서 출력해줘."

# 초기 상태 구성
initial_state = {
    "messages": [
        SystemMessage(content=system_prompt),
        HumanMessage(content=initial_user_query)
    ]
}

# 스트리밍 방식으로 그래프 실행 및 결과 확인
for event in app.stream(initial_state, {"recursion_limit": 10}):
    for key, value in event.items():
        print(f"--- Output from node: {key} ---")
        print(value)
        print("---")
```

### 5. 결론
이 가이드에서는 기존 CodeAct 프로젝트의 핵심 상호작용 로직을 LangGraph를 사용하여 재구성하는 방법을 단계별로 살펴보았습니다. `demo.py`의 절차적 `while` 루프를 `StateGraph`의 선언적 노드와 엣지 구조로 변환함으로써, 에이전트의 워크플로우가 더 명확해지고 각 구성 요소(상태, 에이전트, 도구)의 책임이 분리되어 유지보수와 확장이 용이해졌습니다. 이 구조를 기반으로 더 복잡한 도구를 추가하거나 에러 핸들링 로직을 정교하게 다듬는 등 다양한 개선을 진행할 수 있습니다.
