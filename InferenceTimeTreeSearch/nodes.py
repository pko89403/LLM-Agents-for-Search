
'''
ITTS(Inference-time Tree Search) 에이전트의 핵심 로직을 구성하는 노드들을 정의합니다.
LLM 호출 및 응답 파싱, 목표 달성 확인 로직이 포함된 최종 버전입니다.
'''
import heapq
import re
from typing import List, Dict, Any
from collections import Counter

from state import SearchState, PrioritizedItem, Observation
from tools import Action, transition, ActType
from llm_utils import generate_text
from prompt_utils import construct_prompt, construct_value_prompt
from prompt import PROMPT

# --- ITTS 헬퍼 함수 ---

class Frontier:
    """탐색할 상태들을 관리하는 우선순위 큐"""
    def __init__(self):
        self._heap: List[PrioritizedItem] = []

    def push(self, score: float, state: Dict[str, Any]):
        heapq.heappush(self._heap, PrioritizedItem(-score, state))

    def pop(self) -> tuple[Dict[str, Any], float]:
        item = heapq.heappop(self._heap)
        return item.state, -item.priority

    def __len__(self): return len(self._heap)
    def is_empty(self): return len(self._heap) == 0

def value_function(state: Dict[str, Any]) -> float:
    """LLM을 호출하여 주어진 상태의 가치를 CoT 추론 기반으로 평가합니다."""
    observation: Observation = state.get("observation")
    goal: str = state.get("goal", "")

    if not observation: return 0.0

    clickables = observation.available_actions.get('clickables', {})
    if isinstance(clickables, dict):
        available_actions_list = list(clickables.keys())[:10]
    else:
        # If it's already a sequence (list/tuple/set), coerce to list and slice
        available_actions_list = list(clickables)[:10]

    obs_summary = {
        'query': observation.query,
        'results': [{'id': p.id, 'title': p.title, 'price': p.price} for p in observation.results[:5]],
        'available_actions': available_actions_list,
        'cart': observation.cart # 장바구니 정보 추가
    }

    prompt = construct_value_prompt(
        observation=str(obs_summary),
        objective=goal
    )

    scores = []
    num_samples = state['branching'] # 가치 함수 평가를 위한 샘플 수
    for i in range(num_samples):
        response = generate_text(prompt, temperature=0.7) # 가치 함수는 일관된 응답이 중요하므로 낮은 temperature
        # print(f"가치 함수 LLM 응답 (샘플 {i+1}/{num_samples}):\n{response}") # 디버깅용
        score_match = re.search(r"Final Score: (\d+\.?\d*)", response)
        if score_match:
            try:
                score = float(score_match.group(1))
                scores.append(max(0.0, min(1.0, score))) # 0.0 ~ 1.0 범위로 제한
            except ValueError:
                pass

    if not scores:
        # LLM 응답 파싱 실패 시 기본값 또는 규칙 기반 점수 반환
        # 임시로 0.0 반환, 실제로는 더 정교한 폴백 필요
        return 0.0

    # 여러 샘플의 평균 점수 반환 (self-consistency)
    avg_score = sum(scores) / len(scores)
    print(f"가치 함수 최종 점수: {avg_score:.4f}")
    return avg_score

def is_goal_reached(observation: Observation, goal: str) -> bool:
    """현재 관찰 상태가 최종 목표를 달성했는지 확인합니다."""
    if not observation:
        return False

    goal_lower = goal.lower()

    # 1. 목표 키워드 추출 (간단한 파싱)
    # 예: "Find a durable camera under $100" -> ["durable", "camera", "$100"]
    # 예: "I need a pair of men's walking shoes, size 10, brand 'Nike'" -> ["men's", "walking", "shoes", "size 10", "nike"]
    keywords = re.findall(r'''\b\w+\b|'\w+'|\$\d+|\d+gb|\d+inch''', goal_lower)
    # Remove common stop words or irrelevant terms for product matching
    keywords = [k for k in keywords if k not in ['find', 'a', 'an', 'the', 'under', 'with', 'at', 'least', 'of', 'ram', 'cheapest', 'need', 'pair', 'brand', 'buy']]

    # 2. 검색 결과에서 목표 상품 확인
    if observation.results:
        for product in observation.results:
            product_title_lower = product.title.lower()

            # 모든 키워드가 상품 제목에 포함되어 있는지 확인
            all_keywords_match = all(k in product_title_lower for k in keywords if not k.startswith(
      '$') and not k.endswith('gb') and not k.endswith('inch'))

            # 가격 조건 확인
            price_match = True
            price_limit_match = re.search(r'\$(\d+)', goal_lower)
            if price_limit_match:
                price_limit = float(price_limit_match.group(1))
                if product.price > price_limit:
                    price_match = False

            if all_keywords_match and price_match:
                # If a product matching keywords and price is found, and 'Add to Cart' or 'Buy Now' is available
                # This implies the user has found the product and is ready to purchase.
                if any(action.lower() in ['add to cart', 'buy now'] for action in observation.available_actions.get('clickables', {}).keys()):
                    return True

    # 3. 장바구니에서 목표 상품 확인
    if observation.cart:
        for cart_item in observation.cart:
            cart_item_title_lower = cart_item.title.lower()
            all_keywords_match = all(k in cart_item_title_lower for k in keywords if not
      k.startswith('$') and not k.endswith('gb') and not k.endswith('inch'))
            if all_keywords_match:
                return True

    return False

def parse_llm_action(response: str) -> List[Action]:
    """LLM의 응답에서 액션을 파싱합니다."""
    actions = []
    answer_phrase = PROMPT['meta_data']['answer_phrase']
    action_splitter = PROMPT['meta_data']['action_splitter']

    if answer_phrase in response:
        response = response.split(answer_phrase)[-1]

    action_blocks = response.strip().split(action_splitter)
    raw_action = action_blocks[1] if len(action_blocks) > 1 else action_blocks[0]

    # Corrected regex patterns
    search_match = re.search(r"search\[['\"](.*?)['\"]\]", raw_action)
    choose_match = re.search(r"choose\[['\"](.*?)['\"]\]", raw_action)
    stop_match = re.search(r"stop\[['\"](.*?)['\"]\]", raw_action)

    if search_match:
        actions.append(Action(type=ActType.SEARCH, parameter=search_match.group(1)))
    elif choose_match:
        actions.append(Action(type=ActType.CHOOSE, parameter=choose_match.group(1)))
    elif stop_match:
        print(f"정지 액션 발견: {stop_match.group(1)}")

    return actions

def propose_actions(state: Dict[str, Any]) -> List[Action]:
    """LLM을 호출하여 행동 후보를 제안받습니다."""
    observation: Observation = state.get("observation")
    if not observation: return []

    clickables = observation.available_actions.get('clickables', {})
    if isinstance(clickables, dict):
        available_actions_list = list(clickables.keys())[:10]
    else:
        # If it's already a sequence (list/tuple/set), coerce to list and slice
        available_actions_list = list(clickables)[:10]

    obs_summary = {
        'query': observation.query,
        'results': [{'id': p.id, 'title': p.title, 'price': p.price} for p in observation.results[:5]],
        'available_actions': available_actions_list
    }

    prompt = construct_prompt(
        observation=str(obs_summary),
        objective=state['goal'],
        previous_action=state['action_history'][-1] if state['action_history'] else "None"
    )

    # 20번 샘플링하여 액션 후보 생성
    candidate_actions = []
    num_samples = state['branching']
    for i in range(num_samples):
        response = generate_text(prompt, temperature=1.0, top_p=0.95) # 핵 샘플링 적용
        # print(f"LLM 응답 (샘플 {i+1}/{num_samples}):\n{response}") # 디버깅용
        parsed_actions = parse_llm_action(response)
        if parsed_actions:
            candidate_actions.extend(parsed_actions)

    if not candidate_actions:
        return []

    # 액션 빈도 계산 및 상위 5개 선택
    action_counts = Counter(candidate_actions)
    # Counter의 key는 Action 객체이므로, 직접 비교 가능
    most_common_actions = [action for action, count in action_counts.most_common(5)]

    return most_common_actions

# --- LangGraph 노드 정의 ---

def initialize_state(state: SearchState) -> Dict[str, Any]:
    """에이전트의 첫 상태를 초기화합니다."""
    from tools import webshop_client
    print("---" + "- 1. 초기 상태 설정 ---")
    initial_observation = webshop_client.reset()

    initial_agent_state = {
        "goal": state['goal'], "max_steps": state['max_steps'], "search_counter": 0,
        "branching": state['branching'], "budget": state['budget'],
        "observation": initial_observation, "action_history": [], "best_score": -1.0,
        "best_state": None, "done": False, "final_answer": None,
    }

    frontier = Frontier()
    initial_score = value_function(initial_agent_state)
    frontier.push(initial_score, initial_agent_state)
    print(f"초기 상태 점수: {initial_score:.4f}")

    return {"frontier": frontier, "search_counter": 1, "best_score": initial_score, "best_state": initial_agent_state}

def expand_frontier(state: SearchState) -> Dict[str, Any]:
    """프론티어를 확장하는 핵심 탐색 루프입니다."""
    print(f"\n---" + f"- {state['search_counter']}. 프론티어 확장 ---")
    frontier: Frontier = state['frontier']
    if frontier.is_empty():
        print("프론티어가 비어있어 탐색을 종료합니다.")
        return {"done": True}

    current_state, score = frontier.pop()
    print(f"현재 상태 (점수: {score:.4f}), 이전 액션: {current_state['action_history'][-1:] if current_state['action_history'] else '없음'}")

    best_score, best_state = state.get('best_score', -1.0), state.get('best_state')
    if score > best_score:
        print(f"최고 점수 갱신: {best_score:.4f} -> {score:.4f}")
        best_score, best_state = score, current_state

    if is_goal_reached(current_state['observation'], current_state['goal']):
        print("목표 달성! 탐색을 종료합니다.")
        return {"done": True, "final_answer": "목표 상품을 찾았습니다.", "best_state": best_state, "best_score": best_score}

    actions = propose_actions(current_state)
    if not actions:
        print("제안된 행동이 없어 현재 분기 탐색을 중단합니다.")
        # 이 경우, 프론티어에 아무것도 추가하지 않고 현재 상태를 그대로 반환하여 다음 루프를 돌게 합니다.
        return {"frontier": frontier, "best_score": best_score, "best_state": best_state, "search_counter": state.get("search_counter", 0) + 1}

    print(f"제안된 행동 후보: {[str(a) for a in actions]}")

    for action in actions:
        next_state_update = transition(current_state, action)
        next_state = {**current_state, **next_state_update}
        new_score = value_function(next_state)
        print(f"  - 행동 '{action}' -> 새 상태 점수: {new_score:.4f}")
        frontier.push(new_score, next_state)

    return {
        "frontier": frontier,
        "search_counter": state.get("search_counter", 0) + 1,
        "best_score": best_score,
        "best_state": best_state,
    }

def check_finish_condition(state: SearchState) -> str:
    """탐색 종료 조건을 확인합니다."""
    if state.get("done"): return "finish"
    if state.get("search_counter", 0) > state.get("max_steps", 10): return "finish"
    if state.get("search_counter", 0) > state.get("budget", 64): return "finish"
    return "continue"


    """LLM의 응답에서 액션을 파싱합니다."""
    actions = []
    answer_phrase = PROMPT['meta_data']['answer_phrase']
    action_splitter = PROMPT['meta_data']['action_splitter']

    if answer_phrase in response:
        response = response.split(answer_phrase)[-1]

    action_blocks = response.strip().split(action_splitter)
    raw_action = action_blocks[1] if len(action_blocks) > 1 else action_blocks[0]

    # Corrected regex patterns
    search_match = re.search(r"search\[['\"](.*?)['\"]\]", raw_action)
    choose_match = re.search(r"choose\[['\"](.*?)['\"]\]", raw_action)
    stop_match = re.search(r"stop\[['\"](.*?)['\"]\]", raw_action)

    if search_match:
        actions.append(Action(type=ActType.SEARCH, parameter=search_match.group(1)))
    elif choose_match:
        actions.append(Action(type=ActType.CHOOSE, parameter=choose_match.group(1)))
    elif stop_match:
        print(f"정지 액션 발견: {stop_match.group(1)}")

    return actions
