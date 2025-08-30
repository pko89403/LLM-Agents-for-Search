'''
LLM에 전달할 프롬프트를 구성하는 유틸리티 함수.
'''
from prompt import PROMPT

def construct_prompt(observation: str, objective: str, previous_action: str) -> str:
    """
    퓨샷 프롬프트를 구성합니다.
    """
    # Intro + Examples
    prompt = PROMPT['intro']
    prompt += "\n---\n\n"
    for obs, act in PROMPT['examples']:
        prompt += f"{obs}\n{act}\n\n---\n\n"
    
    # Current situation
    prompt += PROMPT['template'].format(
        observation=observation,
        objective=objective,
        previous_action=previous_action
    )
    return prompt

def construct_value_prompt(observation: str, objective: str) -> str:
    """
    가치 함수 평가를 위한 프롬프트를 구성합니다.
    """
    # Intro + Examples
    prompt = PROMPT['value_prompt']['intro']
    prompt += "\n---\n\n"
    for obs, score_reasoning in PROMPT['value_prompt']['examples']:
        prompt += f"{obs}\n{score_reasoning}\n\n---\n\n"
    
    # Current situation
    prompt += PROMPT['value_prompt']['template'].format(
        observation=observation,
        objective=objective
    )
    return prompt

