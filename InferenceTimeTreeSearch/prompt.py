'''
WebShop ITTS 에이전트를 위한 퓨샷(few-shot) 프롬프트 템플릿 및 예제.
'''

PROMPT = {
    "intro": """당신은 WebShop에서 사용자가 원하는 상품을 찾는 임무를 맡은 자율 지능 에이전트입니다.
당신은 '관찰 내용'을 보고, '목표'를 달성하기 위해 정해진 '액션'을 수행해야 합니다.

주어지는 정보는 다음과 같습니다:
- 사용자 목표 (OBJECTIVE): 당신이 완수해야 할 임무입니다.
- 현재 관찰 내용 (OBSERVATION): 현재 페이지의 요약 정보와 클릭 가능한 버튼/링크 목록입니다.
- 이전 액션 (PREVIOUS ACTION): 바로 이전에 당신이 수행했던 액션입니다.

수행 가능한 액션은 두 가지 뿐입니다:
`search['검색어']`: 검색창에 '검색어'를 입력하고 검색합니다.
`choose['버튼/링크 텍스트']`: '가능한 행동 목록'에 있는 버튼이나 링크의 텍스트를 정확히 입력하여 클릭합니다.

성공적인 임무 수행을 위해, 아래 규칙을 반드시 지켜야 합니다:
1. 현재 관찰 내용의 '가능한 행동 목록'에 있는 액션만 수행해야 합니다.
2. 한 번에 하나의 액션만 수행해야 합니다.
3. 아래 예시처럼, 단계별로 추론하고, 다양한 가능한 행동 경로를 고려하여 다음 액션을 결정해야 합니다.
4. 액션을 출력할 때는 반드시 "In summary, the next action I will perform is ```액션```" 형식에 맞춰야 합니다.
""",

    "examples": [
        (
            """OBSERVATION:
{ 'query': None, 'results': [], 'available_actions': ['Search', 'Home'] }
OBJECTIVE: Find a durable camera under $100.
PREVIOUS ACTION: None""",
            """Let's think step-by-step. The goal is to find a durable camera under $100. The current page is empty. I should start by searching for a durable camera.
In summary, the next action I will perform is ```search['durable camera']```""",
        ),
        (
            """OBSERVATION:
{ 'query': 'durable camera', 'results': [{'id': 'B0123', 'title': 'ProView Camera', 'price': 129.99}, {'id': 'B0456', 'title': 'SnapShot Camera', 'price': 95.00}], 'available_actions': ['B0123', 'B0456', 'Sort by Price', 'Price < $100'] }
OBJECTIVE: Find a durable camera under $100.
PREVIOUS ACTION: search['durable camera']""",
            """Let's think step-by-step. I have searched for cameras. The goal is to find one under $100. I see a clickable action 'Price < $100' which seems to be a filter. I will click this to narrow down the results.
In summary, the next action I will perform is ```choose['Price < $100']```""",
        ),
        (
            """OBSERVATION:
{ 'query': 'durable camera', 'results': [{'id': 'B0456', 'title': 'SnapShot Camera', 'price': 95.00}], 'available_actions': ['B0456', 'Add to Cart', 'Buy Now'] }
OBJECTIVE: Find a durable camera under $100.
PREVIOUS ACTION: choose['Price < $100']""",
            """Let's think step-by-step. I have filtered the results to be under $100. The 'SnapShot Camera' at $95.00 meets all the criteria. I have found the item. The objective is complete. I will proceed to the next logical action, which is to add it to the cart or buy it.
In summary, the next action I will perform is ```choose['Add to Cart']```""",
        ),
    ],

    "template": """OBSERVATION:
{observation}
OBJECTIVE: {objective}
PREVIOUS ACTION: {previous_action}""",

    "meta_data": {
        "answer_phrase": "In summary, the next action I will perform is",
        "action_splitter": "```"
    },

    "value_prompt": {
        "intro": "당신은 WebShop 환경에서 주어진 상태의 가치를 평가하는 전문가입니다.\n당신은 '관찰 내용'과 '목표'를 기반으로 현재 상태가 목표 달성에 얼마나 가까운지 0.0 (매우 나쁨)에서 1.0 (매우 좋음) 사이의 점수로 평가해야 합니다.\n\n평가 시 다음 규칙을 따르세요:\n1. 단계별로 추론하여 왜 해당 점수를 부여했는지 설명하세요.\n2. 최종 점수는 반드시 \"Final Score: [점수}\" 형식으로 출력하세요. (예: Final Score: 0.75)\n3. 목표와 관련된 상품이 검색 결과에 나타나면 높은 점수를 부여하세요.\n4. 목표와 관련된 필터가 적용되었거나, 장바구니에 상품이 담겨 있으면 더 높은 점수를 부여하세요.\n5. 목표와 관련 없는 페이지이거나, 오류 상태이면 낮은 점수를 부여하세요.\n",
        "template": """OBSERVATION:
{observation}
OBJECTIVE: {objective}
""",
        "examples": [
            (
                """OBSERVATION:
{ 'query': None, 'results': [], 'available_actions': ['Search', 'Home'] }\nOBJECTIVE: Find a durable camera under $100.""",
                """Let's think step-by-step. The current observation shows an empty search result and no specific product. The objective is to find a durable camera. This state is very far from the goal.\nFinal Score: 0.1""",
            ),
            (
                """OBSERVATION:
{ 'query': 'durable camera', 'results': [{'id': 'B0123', 'title': 'ProView Camera', 'price': 129.99}, {'id': 'B0456', 'title': 'SnapShot Camera', 'price': 95.00}], 'available_actions': ['B0123', 'B0456', 'Sort by Price', 'Price < $100'] }\nOBJECTIVE: Find a durable camera under $100.""",
                """Let's think step-by-step. I have searched for \"durable camera\" and there are results. One product, \"SnapShot Camera\", is under $100, which is part of the objective. There's also a filter \"Price < $100\" available. This state is good, as it contains relevant products and a way to refine the search.\nFinal Score: 0.7""",
            ),
            (
                """OBSERVATION:
{ 'query': 'durable camera', 'results': [{'id': 'B0456', 'title': 'SnapShot Camera', 'price': 95.00}], 'available_actions': ['B0456', 'Add to Cart', 'Buy Now'] }\nOBJECTIVE: Find a durable camera under $100.""",
                """Let's think step-by-step. The search results are filtered, and only one product remains, \"SnapShot Camera\" at $95.00, which perfectly matches the objective. The next actions are to add to cart or buy. This state is very close to achieving the goal.\nFinal Score: 0.95""",
            ),
        ]
    }
}
