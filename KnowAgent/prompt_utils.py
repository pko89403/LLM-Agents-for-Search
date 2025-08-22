from __future__ import annotations

from typing import Dict

import tiktoken
from langchain.prompts import PromptTemplate

from prompt import KNOWAGENT_EXAMPLE, KNOWAGENT_INSTRUCTION


def truncate_scratchpad(scratchpad: str, n_tokens: int, tokenizer=None) -> str:
    """스크래치패드를 주어진 토큰 수에 맞춰 자릅니다.
    가장 긴 Observation부터 제거하여 스크래치패드 길이를 줄입니다.
    """
    if tokenizer is None:
        try:
            tokenizer = tiktoken.encoding_for_model(
                "gpt-4"
            )  # Default to gpt-4 tokenizer
        except Exception:
            # Fallback if tiktoken cannot find the model or is not installed.
            # In a real scenario, you might want to log this or raise a more specific error.
            return scratchpad  # Cannot truncate without a tokenizer

    lines = scratchpad.split("\n")
    # Observation 라인만 필터링
    observations = [line for line in lines if line.startswith("Observation")]
    # Observation 라인을 토큰 길이에 따라 정렬 (긴 것부터)
    observations_by_tokens = sorted(
        observations, key=lambda x: len(tokenizer.encode(x)), reverse=True
    )

    current_tokens = len(tokenizer.encode("\n".join(lines)))

    # 스크래치패드 길이가 n_tokens보다 클 경우
    while current_tokens > n_tokens and observations_by_tokens:
        largest_observation = observations_by_tokens.pop(0)  # 가장 긴 Observation 제거
        # 해당 Observation 라인을 찾아서 [truncated] 표시로 대체
        # 정확한 라인을 찾기 위해 원본 lines에서 인덱스를 찾음
        for i, line in enumerate(lines):
            if line == largest_observation:
                lines[i] = (
                    largest_observation.split(":")[0]
                    + ": [truncated wikipedia excerpt]"
                )
                break
        current_tokens = len(tokenizer.encode("\n".join(lines)))  # 토큰 재계산

    return "\n".join(lines)


def build_prompt(state: Dict, stage: str) -> str:
    """KnowAgent 프롬프트 템플릿에 현재 상태를 채워 최종 프롬프트 문자열을 생성합니다.
    stage 매개변수에 따라 ActionPath, Thought, Action 중 어떤 부분을 생성할지 지시합니다.
    """
    scratchpad = state.get("scratchpad", "")
    question = state.get("question", "")
    step = state.get("step", 1)

    # 기본 프롬프트 구성 (지시사항, 예제, 질문, 현재 스크래치패드)
    # KNOWAGENT_INSTRUCTION은 이미 {examples}, {question}, {scratchpad} 플레이스홀더를 포함하고 있음
    base_prompt_template = PromptTemplate(
        input_variables=["examples", "question", "scratchpad"],
        template=KNOWAGENT_INSTRUCTION,
    )

    base_prompt = base_prompt_template.format(
        examples=KNOWAGENT_EXAMPLE,
        question=question,
        scratchpad=scratchpad,
    )

    # stage에 따라 추가적인 지시사항을 덧붙임
    if stage == "action_path":
        # Explicitly ask for ActionPath only
        final_prompt = f"{base_prompt}\n\nGenerate only the ActionPath for step {step}. Do not generate Thought or Action.\nActionPath {step}:"
    elif stage == "thought":
        # Explicitly ask for Thought only
        final_prompt = f"{base_prompt}\n\nGenerate only the Thought for step {step}. Do not generate ActionPath or Action.\nThought {step}:"
    elif stage == "action":
        # Explicitly ask for Action only
        final_prompt = f"{base_prompt}\n\nGenerate only the Action for step {step}. Do not generate ActionPath or Thought.\nAction {step}:"
    else:
        raise ValueError(f"Unknown stage for prompt generation: {stage}")

    return final_prompt



