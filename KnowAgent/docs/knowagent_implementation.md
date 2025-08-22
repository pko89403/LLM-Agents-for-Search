# KnowAgentHotpotQA 구현 상세 설명

이 문서는 `KnowAgentHotpotQA` 에이전트의 핵심 구현, 특히 프롬프트 구성(`_build_agent_prompt`)과 실행 흐름(`forward`)에 대해 기술합니다.

## 클래스 구조

`KnowAgentHotpotQA` 클래스는 `BaseAgent` 클래스를 상속받아 구현되었습니다. `BaseAgent`는 에이전트의 기본적인 실행 루프(`run`, `step`), 상태 관리(`scratchpad`, `step_n`), 도구 연동(`docstore`, `bingsearch`) 등 공통 기능을 제공하며, `KnowAgentHotpotQA`는 HotpotQA 태스크에 특화된 프롬프트와 실행 로직을 정의합니다.

```python

# hotpotqa_run/agent_arch.py


classKnowAgentHotpotQA(BaseAgent):

def__init__(self, ...):

super().__init__(...)

self.examples =KNOWAGENT_EXAMPLE

self.agent_prompt = knowagent_prompt

self.name ="KnowAgentHotpotQA"

```

## 프롬프트 구성: `_build_agent_prompt()`

이 메서드는 LLM에 전달할 최종 프롬프트를 동적으로 생성하는 역할을 합니다. `BaseAgent`의 `prompt_agent` 메서드 내부에서 호출됩니다.

```python

# hotpotqa_run/agent_arch.py


def_build_agent_prompt(self) ->str:

returnself.agent_prompt.format(

examples=self.examples,

question=self.question,

scratchpad=self.scratchpad)

```

프롬프트는 `hotpotqa_run/pre_prompt.py`에 정의된 `knowagent_prompt` 템플릿을 사용하며, 세 가지 주요 구성 요소로 이루어집니다.

1.**지시사항 및 예제 (`examples`)**:

*`hotpotqa_run/pre_prompt.py`의 `KNOWAGENT_INSTRUCTION` 템플릿은 에이전트의 목표, 사용 가능한 Action의 종류와 규칙(Action Knowledge Graph)을 정의합니다.

*`hotpotqa_run/fewshots.py`의 `KNOWAGENT_EXAMPLE`은 LLM에게 원하는 결과물의 형식을 보여주는 몇 가지 구체적인 질의-응답 예시(Few-shot Examples)를 제공합니다. 이는 LLM이 `Thought`, `Action`, `Observation`의 흐름을 학습하도록 돕습니다.

2.**질문 (`question`)**:

* 에이전트가 현재 해결해야 할 실제 질문입니다.

3.**스크래치패드 (`scratchpad`)**:

* 에이전트의 "작업 기억 공간"입니다. 현재 질문에 대한 이전 단계들의 `ActionPath`, `Thought`, `Action`, `Observation` 로그가 순차적으로 기록됩니다.
* LLM은 이 `scratchpad` 내용을 보고 현재까지의 진행 상황을 파악하여 다음 단계를 추론합니다.

결과적으로 LLM은 상세한 지시사항과 예제를 보고, 현재 질문과 지금까지의 작업 내역(`scratchpad`)을 바탕으로 다음 `ActionPath`, `Thought`, `Action`을 순서대로 생성하게 됩니다.

## 실행 흐름: `forward()`

`forward` 메서드는 에이전트의 단일 추론 단계(single reasoning step)를 정의합니다. `BaseAgent`의 `step` 메서드 내에서 호출되며, LLM과의 상호작용을 통해 다음 행동을 결정합니다.

```python

# hotpotqa_run/agent_arch.py


defforward(self):

self._actionpath()

self._think()

    action_type, argument =self._action()

return action_type, argument

```

`forward` 메서드의 실행 순서는 다음과 같습니다.

1.**`self._actionpath()` 호출**:

*`BaseAgent`에 정의된 이 메서드는 `_build_agent_prompt()`로 생성된 프롬프트를 LLM에 전달하여 현재까지의 Action 경로인 `ActionPath`를 생성하도록 요청합니다. (예: `Start -> Retrieve`)

* 생성된 `ActionPath`는 `scratchpad`에 추가됩니다.

2.**`self._think()` 호출**:

* 업데이트된 `scratchpad`를 포함한 프롬프트를 다시 LLM에 전달하여, 다음 행동을 결정하기 위한 논리적 추론 과정인 `Thought`를 생성하도록 요청합니다.
* 생성된 `Thought` 역시 `scratchpad`에 추가됩니다.

3.**`self._action()` 호출**:

* 다시 업데이트된 `scratchpad`를 포함한 프롬프트를 LLM에 전달하여, 최종적으로 수행할 `Action`과 그에 필요한 인자(argument)를 생성하도록 요청합니다. (예: `Lookup[Ariel's role]`)
* 생성된 `Action`을 `scratchpad`에 추가하고, `action_type`과 `argument`를 파싱하여 반환합니다.

`forward` 메서드가 `action_type`과 `argument`를 반환하면, `BaseAgent`의 `step` 메서드는 이를 받아 `Search`, `Retrieve`, `Lookup`, `Finish` 등 실제 도구를 실행하고 그 결과를 `Observation`으로 `scratchpad`에 기록합니다. 이 전체 과정이 `run` 메서드에 의해 `is_halted()` 또는 `is_finished()`가 `True`가 될 때까지 반복됩니다.
