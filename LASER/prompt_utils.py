# -*- coding: utf-8 -*-
"""LLM 프롬프트 구성 및 스크래치패드 관리 유틸리티를 정의합니다."""

from __future__ import annotations

import logging
import tiktoken
from typing import Dict, List, Any

from langchain_core.messages import SystemMessage, HumanMessage

from prompt import ( # prompt.py에서 필요한 프롬프트 템플릿 임포트
    INDIVIDUAL_SYSTEM_PROMPT,
    SEARCH_STATE_PROMPT_ADDON, SEARCH_STATE_GUIDE,
    SELECT_STATE_PROMPT_ADDON, SELECT_STATE_GUIDE,
    VERIFY_STATE_PROMPT_ADDON, VERIFY_STATE_GUIDE,
    SCORE_SYSTEM_PROMPT, SCORE_HUMAN_PROMPT_TEMPLATE,
    MAPPING_ACTION_SYSTEM_PROMPT, MAPPING_ACTION_HUMAN_PROMPT,
    FEEDBACK_SYSTEM_PROMPT, FEEDBACK_HUMAN_PROMPT,
    RETHINK_SYSTEM_PROMPT, RETHINK_HUMAN_PROMPT,
    MANAGER_SYSTEM_PROMPT, MANAGER_HUMAN_PROMPT
)
from parsing_utils import _parse_target_instruction # _parse_target_instruction 함수를 임포트


def truncate_scratchpad(scratchpad: str, n_tokens: int = 16000, tokenizer_name: str = "cl100k_base") -> str:
    """스크래치패드를 주어진 토큰 수에 맞춰 자릅니다."""
    try:
        tokenizer = tiktoken.get_encoding(tokenizer_name)
    except Exception:
        tokenizer = tiktoken.get_encoding("gpt2")

    tokens = tokenizer.encode(scratchpad)

    if len(tokens) > n_tokens:
        tokens = tokens[-n_tokens:]
        return tokenizer.decode(tokens)

    return scratchpad


def build_prompt(
    state: Dict[str, Any],
    parsed_obs: Dict[str, Any],
    current_laser_state: str,
    prompt_type: str = "default",
    rationale: Optional[str] = None
) -> List[Any]:
    """LASER 에이전트의 상태 및 목적에 따라 프롬프트를 구성합니다."""

    # prompt_type에 따라 분기
    if prompt_type == "mapping":
        # 자가 교정을 위한 액션 매핑 프롬프트
        observation = state.get("obs", "")
        system_message_content = MAPPING_ACTION_SYSTEM_PROMPT.format(
            observation=observation,
            rationale=rationale or "No rationale provided."
        )
        messages = [
            SystemMessage(content=system_message_content),
            HumanMessage(content=MAPPING_ACTION_HUMAN_PROMPT),
        ]
        return messages

    # 기본 프롬프트 (기존 로직)
    elif prompt_type == "default":
        user_instruction = state.get("user_instruction", "")
        current_observation = state.get("obs", "")
        action_history = state.get("action_history", [])
        thought_history = state.get("thought_history", [])

        system_message_content = ""
        human_message_content = f"Current observation:\n{current_observation}"

        # 과거 기록 추가
        if action_history or thought_history:
            history_str = "\nHistory:\n"
            for i in range(max(len(action_history), len(thought_history))):
                if i < len(thought_history):
                    history_str += f"Rationale{i}: {thought_history[i]}\n"
                if i < len(action_history):
                    history_str += f"Action{i}: {action_history[i]}\n"
            human_message_content += history_str

        # 상태별 프롬프트 구성
        if current_laser_state == "Search":
            system_message_content = INDIVIDUAL_SYSTEM_PROMPT % (
                "shopping", "find the right item", "web navigation session",
                SEARCH_STATE_PROMPT_ADDON.format(the_user_instruction=user_instruction),
                SEARCH_STATE_GUIDE
            )
            human_message_content += f"\nInstruction: {user_instruction}"

        elif current_laser_state == "Result":
            page_info = parsed_obs.get("page_info", {})
            items_str = ""
            for item in parsed_obs.get("items", []):
                items_str += f"[button] {item.get('item_id', '')} [button_]\n{item.get('name', '')}\n{item.get('price_str', '')}\n"
            if parsed_obs.get("items"):
                items_str += "{More items...}\n"

            select_prompt_addon_formatted = SELECT_STATE_PROMPT_ADDON.format(
                user_instruction=user_instruction,
                current_page_number=page_info.get("current_page", "N/A"),
                total_number_of_results=page_info.get("total_results", "N/A"),
            ) + "\n" + items_str

            system_message_content = INDIVIDUAL_SYSTEM_PROMPT % (
                "shopping", "find the right item", "web navigation session",
                select_prompt_addon_formatted,
                SELECT_STATE_GUIDE
            )
            human_message_content += f"\nInstruction: {user_instruction}"

        elif current_laser_state == "Item":
            target_info = _parse_target_instruction(user_instruction)
            target_keywords = ", ".join(target_info.get("keywords", []))
            target_max_price = f"${target_info['max_price']}" if target_info.get("max_price") is not None else "None"
            raw_obs = parsed_obs.get("raw_obs", "")
            item_name_and_details = ""
            if raw_obs:
                parts = raw_obs.split("\n[button] Description")
                if len(parts) > 0:
                    sub_parts = parts[0].split("\n")
                    if len(sub_parts) > 0:
                        item_name_and_details = sub_parts[-1]

            verify_prompt_addon_formatted = VERIFY_STATE_PROMPT_ADDON.format(
                user_instruction=user_instruction,
                Customization_type1=list(parsed_obs.get("customizations", {}).keys())[0] if parsed_obs.get("customizations") else "Customization type1",
                option1=list(parsed_obs.get("customizations", {}).values())[0][0] if parsed_obs.get("customizations") else "option1",
                Customization_type2=list(parsed_obs.get("customizations", {}).keys())[1] if len(parsed_obs.get("customizations", {})) > 1 else "Customization type2",
                option2=list(parsed_obs.get("customizations", {}).values())[1][0] if len(parsed_obs.get("customizations", {})) > 1 else "option2",
                Item_name_and_details=item_name_and_details,
                full_description_of_the_item="None",
                full_features_of_the_item="None",
                full_reviews_of_the_item="None",
                keywords_of_the_target_item=target_keywords,
                the_price_of_the_item_should_not_exceed_this=target_max_price,
            )

            if parsed_obs.get("description_viewed") and "description:" in raw_obs:
                verify_prompt_addon_formatted += f"\ndescription: {raw_obs.split('description:')[1].splitlines()[0].strip()}"
            if parsed_obs.get("features_viewed") and "features:" in raw_obs:
                verify_prompt_addon_formatted += f"\nfeatures: {raw_obs.split('features:')[1].splitlines()[0].strip()}"
            if parsed_obs.get("reviews_viewed") and "reviews:" in raw_obs:
                verify_prompt_addon_formatted += f"\nreviews: {raw_obs.split('reviews:')[1].splitlines()[0].strip()}"

            system_message_content = INDIVIDUAL_SYSTEM_PROMPT % (
                "shopping", "find the right item", "web navigation session",
                verify_prompt_addon_formatted,
                VERIFY_STATE_GUIDE
            )
            human_message_content += f"\nInstruction: {user_instruction}"

        else:
            logging.warning(f"알 수 없는 LASER 상태: {current_laser_state}. 기본 프롬프트 사용.")
            system_message_content = "You are a helpful assistant."

        messages = [
            SystemMessage(content=system_message_content),
            HumanMessage(content=human_message_content),
        ]
        return messages

    else:
        logging.error(f"지원하지 않는 프롬프트 타입: {prompt_type}")
        return [
            SystemMessage(content="Error: Invalid prompt type specified."),
            HumanMessage(content="")
        ]


def build_scoring_prompt(item_info: Dict[str, Any], user_instruction: str) -> List[Any]:
    """아이템 스코어링을 위한 프롬프트를 구성합니다."""
    item_description = f"제목: {item_info.get('title', 'N/A')}\n가격: {item_info.get('price', 'N/A')}\n"
    if item_info.get('features_summary'):
        item_description += f"특징: {item_info['features_summary']}\n"
    if item_info.get('reviews_summary'):
        item_description += f"리뷰 요약: {item_info['reviews_summary']}\n"

    messages = [
        SystemMessage(content=SCORE_SYSTEM_PROMPT),
        HumanMessage(content=SCORE_HUMAN_PROMPT_TEMPLATE.format(
            user_instruction=user_instruction,
            item_description=item_description
        )),
    ]
    return messages


def build_feedback_prompt(observation: str, rationale: str, action: str) -> List[Any]:
    """피드백을 위한 프롬프트를 구성합니다."""
    system_message_content = FEEDBACK_SYSTEM_PROMPT.format(
        observation=observation,
        rationale=rationale,
        action=action
    )

    human_message_content = FEEDBACK_HUMAN_PROMPT.format(
        observation=observation,
        rationale=rationale,
        action=action
    )

    messages = [
        SystemMessage(content=system_message_content),
        HumanMessage(content=human_message_content),
    ]
    return messages


def build_rethink_prompt(observation: str, rationale: str, action: str, feedback: str, tool_specs: List[Dict]) -> List[Any]:
    """재고를 위한 프롬프트를 구성합니다."""
    # 사용 가능한 도구들을 문자열로 변환
    tools_description = "\n".join([f"- {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')}" for tool in tool_specs])

    system_message_content = RETHINK_SYSTEM_PROMPT.format(
        observation=observation,
        rationale=rationale,
        action=action,
        feedback=feedback
    ) + f"\n\nAvailable tools:\n{tools_description}"

    human_message_content = RETHINK_HUMAN_PROMPT.format(
        observation=observation,
        rationale=rationale,
        action=action,
        feedback=feedback
    )

    messages = [
        SystemMessage(content=system_message_content),
        HumanMessage(content=human_message_content),
    ]
    return messages


def build_manager_prompt(history: str, observation: str, rationale: str, action: str) -> List[Any]:
    """매니저 피드백을 위한 프롬프트를 구성합니다."""
    system_message_content = MANAGER_SYSTEM_PROMPT.format(
        history=history,
        observation=observation,
        rationale=rationale,
        action=action
    )

    human_message_content = MANAGER_HUMAN_PROMPT.format(
        history=history,
        observation=observation,
        rationale=rationale,
        action=action
    )

    messages = [
        SystemMessage(content=system_message_content),
        HumanMessage(content=human_message_content),
    ]
    return messages
