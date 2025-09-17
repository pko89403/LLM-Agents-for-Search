# LangGraph로 AvaTaR 최적화 프레임워크 구현하기: 상세 설계 가이드

## 1. 소개

**AvaTaR (Optimizing LLM Agents for Tool Usage via Contrastive Reasoning)**는 LLM 에이전트의 도구 사용 능력을 자동으로 최적화하는 프레임워크입니다. 이 가이드는 `avatar/models/avatar.py`에 구현된 복잡한 최적화 로직을 **LangGraph**를 사용하여 명확하고 모듈화된 상태 기반 워크플로우로 재설계하는 방법을 상세히 안내합니다.

**핵심 목표:**
- **관심사 분리**: 'Optimizer'의 흐름 제어 로직과 'Actor/Comparator'의 LLM 호출 로직을 분리합니다.
- **명확성**: `try-except`를 이용한 복잡한 제어 흐름을 LangGraph의 명시적인 노드와 엣지로 변환합니다.
- **모듈성**: 각 최적화 단계를 독립적인 노드로 만들어 재사용과 디버깅을 용이하게 합니다.

## 2. 구조 설계: Orchestrator와 Workers

제안하는 구조는 작업을 지시하는 'Orchestrator'와 실제 작업을 수행하는 'Workers'로 나뉩니다.

- **Workers (in `avatar/models/`)**:
    - `Actor`: 코드 생성을 담당하는 LLM 래퍼 클래스. `initialize`, `improve` 메소드를 가집니다.
    - `Comparator`: 성공/실패 사례를 비교하여 개선 피드백을 생성하는 LLM 래퍼 클래스.
    - `Evaluator`: 생성된 코드를 평가 데이터셋에서 실행하고 성능 지표(MRR 등)를 계산하는 모듈.

- **Orchestrator (in `run_langgraph_optimizer.py`)**:
    - **LangGraph StateGraph**: 전체 최적화 과정을 조율하는 상태 머신. `Actor`, `Comparator`, `Evaluator`를 적시에 호출하여 워크플로우를 진행시킵니다.

---

## 3. 단계별 구현 가이드

### 1단계: 최적화 상태(`OptimizerState`) 상세 정의

`optimize_actions` 메소드에서 사용되는 모든 변수를 포함하는 `TypedDict` 상태를 정의합니다. 이 상태는 그래프의 각 노드를 거치며 업데이트됩니다.

```python
from typing import TypedDict, List, Dict, Any, Optional

class OptimizerState(TypedDict):
    # --- 핵심 데이터 ---
    actor_code: str              # 현재 버전의 get_node_score_dict 함수 코드
    best_actor_code: str         # 현재까지 최고 성능을 보인 코드
    
    # --- 진행 상태 ---
    step: int                    # 현재 반복 횟수
    best_step: int               # 최고 성능을 달성한 step
    patience_counter: int        # 성능 개선이 없는 횟수 (원래 코드의 gap_from_last_improv)
    
    # --- 피드백 및 오류 ---
    feedback: str                # Comparator가 생성한 개선 지침 또는 코드 실행 오류
    is_executable: bool          # 현재 코드가 오류 없이 실행 가능한지 여부
    
    # --- 성능 및 기록 ---
    memory_bank: Dict[str, List] # 'action_performance'와 'supervision_info' 저장
    current_metric: float        # 현재 코드의 전체 평가 점수 (MRR)
    best_metric: float           # 현재까지의 최고 MRR 점수
    
    # --- 입력 및 설정 ---
    qa_dataset: Any              # QA 데이터셋
    train_indices: List[int]     # 훈련용 데이터 인덱스
    val_indices: List[int]       # 검증용 데이터 인덱스
    config: Dict[str, Any]       # n_eval, patience, sel_metric 등 설정값
    
    # --- 로깅 ---
    log_history: List[Dict]      # LLM과의 대화 기록
```

### 2단계: Worker 클래스 정의 (관심사 분리)

LangGraph 노드에서 호출할 `Actor`, `Comparator`, `Evaluator`를 별도의 클래스로 정의합니다. 이는 `avatar.py`에 혼재된 역할들을 명확히 분리하는 과정입니다.

```python
# 위치: avatar/models/actor.py
class Actor:
    def __init__(self, llm_client):
        self.llm = llm_client
        # _get_prompt, _parse_output_to_actions 등 헬퍼 함수를 내장합니다.

    def initialize(self, qa_dataset, train_indices, pattern) -> str:
        # 'initialize_actions' 프롬프트를 생성하고 LLM을 호출하여 초기 코드 생성
        # 원본 코드: _get_prompt(name='initialize_actions', ...) 호출 부분
        prompt = self._get_prompt('initialize_actions', ...)
        output = self.llm(prompt)
        code = self._parse_output_to_actions(output)
        return code

    def improve(self, current_code, feedback, memory_bank_info) -> str:
        # 'improve_actions' 프롬프트를 생성하고 LLM을 호출하여 코드 개선
        # 원본 코드: _get_prompt(name='improve_actions', ...) 호출 부분
        # memory_bank_info는 과거의 성공/실패 사례를 프롬프트에 주입하는 데 사용됩니다.
        prompt = self._get_prompt('improve_actions', feedback_message=feedback, memory_info=memory_bank_info, ...)
        output = self.llm(prompt)
        code = self._parse_output_to_actions(output)
        return code

# 위치: avatar/models/comparator.py
class Comparator:
    def __init__(self, llm_client):
        self.llm = llm_client
        # _get_prompt, construct_pos_neg_queries 등 헬퍼 함수를 내장합니다.

    def generate_feedback(self, code_to_evaluate, eval_results, qa_dataset) -> str:
        # eval_results를 바탕으로 pos/neg 쿼리 구성
        # 원본 코드: construct_pos_neg_queries(...) 호출 부분
        pos_neg_queries = self.construct_pos_neg_queries(eval_results, ...)
        
        # 'comparator' 프롬프트를 생성하고 LLM을 호출하여 피드백 생성
        # 원본 코드: _get_prompt(name='comparator', ...) 호출 부분
        prompt = self._get_prompt('comparator', pos_neg_queries=pos_neg_queries)
        feedback = self.llm(prompt)
        return feedback

# 위치: avatar/evaluators/evaluator.py
class Evaluator:
    def __init__(self, kb, apis):
        # kb, APIs 등 평가에 필요한 의존성 주입
        # _exec_actions_from_output, sequential_eval_actions 등 평가 로직 포함
    
    def run_feedback_eval(self, code: str, qa_dataset, indices: List[int]) -> Dict:
        # 'optimize_actions'의 try 블록 내부처럼, 코드 실행 후 각 쿼리에 대한 소규모 평가 수행
        # 이 노드는 피드백 생성을 위한 데이터를 수집하는 것이 주 목적입니다.
        # 성공 시: {'executable': True, 'results': exec_eval}
        # 실패 시: {'executable': False, 'error': traceback_str}
        try:
            # _exec_actions_from_output 및 소규모 배치 평가 로직
            exec_eval = self._run_small_batch_eval(code, qa_dataset, indices)
            return {'executable': True, 'results': exec_eval}
        except Exception as e:
            return {'executable': False, 'error': str(e)}

    def run_full_eval(self, code: str, qa_dataset, indices: List[int]) -> float:
        # 'eval_action'처럼, 검증셋 전체에 대한 전체 성능(MRR) 측정
        # 이 노드는 현재 코드의 최종 성능을 확정하고 최고 기록과 비교하는 역할입니다.
        # 원본 코드: parallel_eval_actions 또는 sequential_eval_actions 호출 부분
        metrics = self._run_validation_eval(code, qa_dataset, indices)
        return metrics['mrr'] # 또는 설정된 주요 메트릭
```

### 3단계: 그래프 노드(Node) 구현

각 노드는 `OptimizerState`를 인자로 받고, Worker를 호출하여 상태를 업데이트합니다. 이는 `optimize_actions`의 복잡한 제어 흐름을 단계별로 분해합니다.

```python
# 위치: run_langgraph_optimizer.py

# --- 노드 구현 ---

def initialize_actor(state: OptimizerState) -> OptimizerState:
    print("--- 1. ACTOR 초기화 ---")
    actor = Actor(llm_client)
    initial_code = actor.initialize(...)
    state['actor_code'] = initial_code
    state['step'] = 0
    # ... 기타 상태 변수 초기화
    return state

def evaluate_for_feedback(state: OptimizerState) -> OptimizerState:
    print(f"--- {state['step']}.1. 피드백 수집용 평가 ---")
    evaluator = Evaluator(...)
    eval_result = evaluator.run_feedback_eval(state['actor_code'], ...)

    if not eval_result['executable']:
        print("   [!] 코드 실행 실패. 오류 메시지를 피드백으로 사용합니다.")
        state['is_executable'] = False
        state['feedback'] = eval_result['error']
        return state

    print("--- {state['step']}.2. 개선 피드백 생성 (COMPARATOR) ---")
    state['is_executable'] = True
    comparator = Comparator(llm_client)
    feedback = comparator.generate_feedback(state['actor_code'], eval_result['results'], ...)
    state['feedback'] = feedback
    
    # 원본 코드: memory_bank.push('supervison_info', ...)
    # state['memory_bank']['supervision_info'].append(...)
    return state

def evaluate_fully(state: OptimizerState) -> OptimizerState:
    print(f"--- {state['step']}.3. 전체 성능 평가 ---")
    evaluator = Evaluator(...)
    metric = evaluator.run_full_eval(state['actor_code'], ...)
    state['current_metric'] = metric
    
    # 원본 코드: memory_bank.push('action_performance', ...)
    # state['memory_bank']['action_performance'].append((state['actor_code'], metric))

    # 최고 성능 갱신 로직
    if metric > state['best_metric']:
        print(f"   [*] 성능 개선! {state['best_metric']:.4f} -> {metric:.4f}")
        state['best_metric'] = metric
        state['best_actor_code'] = state['actor_code']
        state['best_step'] = state['step']
        state['patience_counter'] = 0
    else:
        print(f"   [-] 성능 개선 없음. (현재: {metric:.4f}, 최고: {state['best_metric']:.4f})")
        state['patience_counter'] += 1
        
    return state

def improve_actor(state: OptimizerState) -> OptimizerState:
    print("--- 4. ACTOR 코드 개선 ---")
    actor = Actor(llm_client)
    
    # 메모리 뱅크에서 과거 코드/성능 정보 추출하여 프롬프트에 활용
    # memory_info = format_memory_for_prompt(state['memory_bank'])
    
    new_code = actor.improve(
        current_code=state['actor_code'],
        feedback=state['feedback'],
        memory_bank_info=memory_info
    )
    
    state['actor_code'] = new_code
    state['step'] += 1
    return state

def reinitialize_actor(state: OptimizerState) -> OptimizerState:
    print(f"--- 성능 개선이 {state['patience_counter']}회 없어 ACTOR를 재초기화합니다 ---")
    # initialize_actor와 동일한 로직 수행, step은 유지
    actor = Actor(llm_client)
    initial_code = actor.initialize(...)
    state['actor_code'] = initial_code
    state['patience_counter'] = 0 # 재초기화 후 patience 초기화
    return state
```

### 4단계: 흐름 제어 (엣지) 및 그래프 조립

`optimize_actions`의 복잡한 `try-except`와 `if-else` 분기들을 명시적인 조건부 엣지로 정의합니다.

```python
from langgraph.graph import StateGraph, END

# --- 조건부 엣지 함수 ---

def check_executability(state: OptimizerState) -> str:
    # evaluate_for_feedback 노드 이후, 코드 실행 가능성에 따라 분기
    # 원본 코드의 try-except 블록에 해당
    if state["is_executable"]:
        # 코드가 성공적으로 실행되면, 전체 성능 평가로 이동
        return "evaluate_fully"
    else:
        # 코드 실행이 실패하면, 평가를 건너뛰고 즉시 코드 개선으로 이동
        return "improve_actor"

def check_should_continue(state: OptimizerState) -> str:
    # improve_actor 노드 이후, 루프 지속/재초기화/종료 여부 결정
    max_steps = state['config']['max_steps']
    patience = state['config']['patience']
    
    if state["step"] >= max_steps:
        print("--- 최대 반복 횟수에 도달하여 종료합니다. ---")
        return END
    if state["patience_counter"] >= patience:
        # 원본 코드의 gap_from_last_improv > patience 조건
        return "reinitialize_actor"
    else:
        return "evaluate_for_feedback" # 다음 최적화 루프 시작

# --- 그래프 조립 ---

workflow = StateGraph(OptimizerState)

# 노드 추가
workflow.add_node("initialize_actor", initialize_actor)
workflow.add_node("evaluate_for_feedback", evaluate_for_feedback)
workflow.add_node("evaluate_fully", evaluate_fully)
workflow.add_node("improve_actor", improve_actor)
workflow.add_node("reinitialize_actor", reinitialize_actor)

# 엣지 연결
workflow.set_entry_point("initialize_actor")
workflow.add_edge("initialize_actor", "evaluate_for_feedback")

workflow.add_conditional_edges(
    "evaluate_for_feedback",
    check_executability,
    {"evaluate_fully": "evaluate_fully", "improve_actor": "improve_actor"}
)

workflow.add_edge("evaluate_fully", "improve_actor")
workflow.add_edge("reinitialize_actor", "evaluate_for_feedback")

workflow.add_conditional_edges(
    "improve_actor",
    check_should_continue,
    {"evaluate_for_feedback": "evaluate_for_feedback", 
     "reinitialize_actor": "reinitialize_actor",
     END: END}
)

# 그래프 컴파일
app = workflow.compile()
```

## 5. 결론

이 상세 설계는 `avatar.py`의 복잡한 최적화 로직을 LangGraph를 통해 **명확하고, 모듈적이며, 확장 가능한** 워크플로우로 전환하는 청사진을 제공합니다. 'Orchestrator'와 'Worker'의 분리는 코드의 가독성과 유지보수성을 크게 향상시키며, 각 최적화 단계의 역할을 명확히 합니다. 이 설계를 기반으로 실제 구현을 진행하면 AvaTaR의 강력한 최적화 능력을 보다 체계적인 구조 위에서 활용할 수 있을 것입니다.
