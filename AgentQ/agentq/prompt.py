"""
프롬프트 템플릿 및 Few-shot 예제
"""

from typing import Dict, List, Optional
from agentq.state import AgentState


# 시스템 프롬프트 템플릿
SYSTEM_PROMPTS = {
    "plan": "You are AgentQ, an advanced AI agent that can interact with web pages.\n\nYour task is to create a step-by-step plan to accomplish the user's objective.\n\nGuidelines:\n- Break down the objective into clear, actionable steps\n- Consider what web interactions might be needed\n- Keep the plan concise but comprehensive\n- Number each step clearly\n\nUser Objective: {objective}\n\nPlease create a detailed plan to accomplish this objective.",

    "thought": "You are AgentQ, an advanced AI agent that can interact with web pages.\nBased on the current context, generate a concise thought, a list of 3-5 next possible commands, and update the plan and status.\n\nContext:\n- Objective: {objective}\n- Current URL: {current_url}\n- Page Title: {page_title}\n- Plan: {plan}\n- Loop: {loop_count}/{max_loops}\n- Previous observation: {observation}\n\nAction grammar:\n- GOTO [URL=<http(s)://...> ]\n- SEARCH [TEXT=<query>]\n- CLICK [ID=<data-agentq-id>]\n- TYPE [ID=<data-agentq-id>] [TEXT=<free text>]\n- SUBMIT [ID=<data-agentq-id>]\n- CLEAR [ID=<data-agentq-id>]\n- SCROLL [UP|DOWN]\n- GET_DOM\n- SCREENSHOT [PATH=<filename.png>]\n- WAIT [SECONDS=<int>]\n- ASK USER HELP [TEXT=<question>]",

    "explanation": "You are AgentQ, an advanced AI agent that can interpret web interaction results.\n\nYou just executed an action and received an observation. Your task is to:\n1. Interpret what happened\n2. Explain the significance of the result\n3. Determine if this brings us closer to the objective\n\nCurrent context:\n- Objective: {objective}\n- Action taken: {action}\n- Observation: {observation}\n\nPlease provide a clear explanation of what happened and its significance.",

    "critique": "You are AgentQ, an advanced AI agent that evaluates task completion.\n\nYour task is to determine if the objective has been accomplished based on the current state.\n\nCurrent context:\n- Objective: {objective}\n- Plan: {plan}\n- Loop count: {loop_count}/{max_loops}\n- Latest explanation: {explanation}\n- Scratchpad: {scratchpad}\n\nEvaluation criteria:\n1. Has the main objective been achieved?\n2. Is there sufficient information to provide a complete answer?\n3. Are we making progress or stuck in a loop?\n4. Should we continue or stop here?\n\nRespond with either:\n- \"CONTINUE\" if more actions are needed\n- \"COMPLETE\" if the objective has been accomplished\n\nProvide your reasoning.",

    "critic": "You are AgentQ's self-critic. Your task is to rank the given candidate commands.\n\nYou are given:\n- Objective: {objective}\n- Current URL: {current_url}\n- Page Title: {page_title}\n- Scratchpad: {scratchpad}\n- Latest observation: {observation}\n\nScoring rules (0.0~1.0):\n- + Progress toward goal, low-risk, minimal detours\n- + Uses visible interactive element IDs when clicking/typing\n- - Dead-ends (login, cookie banners) unless necessary\n- - Irreversible or off-domain navigation without reason"
}


# Few-shot 예제
FEW_SHOT_EXAMPLES = {
    "plan": [
        {
            "objective": "프랑스의 수도가 뭐야?",
            "plan": """1. Google에서 \"프랑스 수도\" 검색\n2. 검색 결과에서 정확한 정보 찾기\n3. 답변 정리하여 사용자에게 제공"""
        },
        {
            "objective": "네이버로 이동해줘",
            "plan": """1. 네이버 웹사이트 URL로 이동 (https://www.naver.com)\n2. 페이지 로딩 확인\n3. 이동 완료 보고"""
        }
    ],

    "thought": [
        {
            "context": "OpenTable home loaded; need to search restaurant",
            "reasoning": "Propose multiple safe next-steps using allowed grammar",
            "action": "Example output:\n\nPLAN:\nSearch for the restaurant then pick a time.\n\nTHOUGHT:\nWe should search first.\n\nCOMMANDS:\n- TYPE [ID=search_input] [TEXT=Cecconi's New York]\n- CLICK [ID=submit_search]\n- GET_DOM\n\nSTATUS:\nCONTINUE"
        }
    ],

    "explanation": [
        {
            "action": "SEARCH: 프랑스 수도",
            "observation": "검색 완료. 결과: 파리(Paris)는 프랑스의 수도이자 최대 도시입니다...",
            "explanation": "Google 검색을 통해 프랑스의 수도가 파리라는 정보를 성공적으로 찾았습니다. 이는 사용자의 질문에 대한 정확한 답변을 제공할 수 있는 충분한 정보입니다."
        }
    ],

    "critique": [
        {
            "objective": "프랑스의 수도가 뭐야?",
            "explanation": "Google 검색을 통해 프랑스의 수도가 파리라는 정보를 찾았습니다.",
            "decision": "COMPLETE",
            "reasoning": "사용자의 질문에 대한 정확한 답변을 얻었으므로 작업이 완료되었습니다."
        }
    ]
}


def get_system_prompt(prompt_type: str, **kwargs) -> str:
    """시스템 프롬프트 가져오기"""
    if prompt_type not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    template = SYSTEM_PROMPTS[prompt_type]
    return template.format(**kwargs)


def get_few_shot_examples(prompt_type: str, num_examples: int = 2) -> str:
    """Few-shot 예제 가져오기"""
    if prompt_type not in FEW_SHOT_EXAMPLES:
        return ""

    examples = FEW_SHOT_EXAMPLES[prompt_type][:num_examples]

    if prompt_type == "plan":
        return "\n\n".join([
            f"Example:\nObjective: {ex['objective']}\nPlan:\n{ex['plan']}"
            for ex in examples
        ])
    elif prompt_type == "thought":
        return "\n\n".join([
            f"Example:\nContext: {ex['context']}\nReasoning: {ex['reasoning']}\nAction: {ex['action']}"
            for ex in examples
        ])
    elif prompt_type == "explanation":
        return "\n\n".join([
            f"Example:\nAction: {ex['action']}\nObservation: {ex['observation']}\nExplanation: {ex['explanation']}"
            for ex in examples
        ])
    elif prompt_type == "critique":
        return "\n\n".join([
            f"Example:\nObjective: {ex['objective']}\nExplanation: {ex['explanation']}\nDecision: {ex['decision']}\nReasoning: {ex['reasoning']}"
            for ex in examples
        ])

    return ""


def build_prompt_with_examples(prompt_type: str, include_examples: bool = True, **kwargs) -> str:
    """예제를 포함한 프롬프트 구성"""
    system_prompt = get_system_prompt(prompt_type, **kwargs)

    if include_examples:
        examples = get_few_shot_examples(prompt_type)
        if examples:
            system_prompt += f"\n\nHere are some examples:\n{examples}\n\nNow, please respond to the current situation:"

    return system_prompt


def format_state_for_prompt(state: AgentState) -> Dict[str, str]:
    """AgentState를 프롬프트용 딕셔너리로 변환"""
    from agentq.state import get_scratchpad_content

    return {
        "objective": state["objective"],
        "plan": state["plan"] or "No plan yet",
        "thought": state["thought"] or "No previous thought",
        "action": str(state["action"]) if state["action"] else "No previous action",
        "observation": state["observation"] or "No previous observation",
        "explanation": state["explanation"] or "No previous explanation",
        "loop_count": str(state["loop_count"]),
        "max_loops": str(state["max_loops"]),
        "scratchpad": get_scratchpad_content(state),
        "current_url": state["current_url"] or "No current URL",
        "page_title": state["page_title"] or "No page title"
    }
