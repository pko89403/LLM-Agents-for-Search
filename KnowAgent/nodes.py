import logging
from typing import Dict

from prompt_utils import (
    build_prompt,
    truncate_scratchpad
)
from tools import (
    lookup_keyword,
    parse_action,
    web_search,
    wikipedia_retrieve
)


def node_decide(
    state: Dict, llm, max_consec_search: int, auto_finish_step: int, context_len: int
) -> Dict:
    """LLM을 호출하여 ActionPath, Thought, Action을 순차적으로 생성하고 scratchpad에 반영합니다.

    각 단계에서 LLM은 이전 단계의 결과와 전체 궤적 기록(scratchpad)에 조건화됩니다.
    모델이 포맷을 따르지 못하는 경우를 대비해 안전한 폴백을 제공합니다:
    - step==1 이고 파싱 실패 시: Search[question]
    - 그 외 파싱 실패 시: Finish[best-effort or Unknown]
    """
    logging.info(f"node_decide 실행 (Step {state.get('step', 1)})...")
    step = state.get("step", 1)
    current_scratchpad = state.get("scratchpad", "")

    # Truncate scratchpad before each LLM call
    truncated_scratchpad = truncate_scratchpad(current_scratchpad, context_len)
    state["scratchpad"] = (
        truncated_scratchpad  # Update state with truncated scratchpad for prompt building
    )

    # 1. Generate ActionPath
    logging.info(f"Step {step}: ActionPath 생성 중...")
    prompt_action_path = build_prompt(state, stage="action_path")
    resp_action_path = llm.invoke(prompt_action_path)
    action_path_text = getattr(resp_action_path, "content", None) or str(
        resp_action_path
    )
    action_path = action_path_text.strip()
    current_scratchpad += f"\nActionPath {step}: {action_path}"
    state["scratchpad"] = current_scratchpad  # Update state for next prompt

    # Truncate again before next LLM call
    truncated_scratchpad = truncate_scratchpad(current_scratchpad, context_len)
    state["scratchpad"] = truncated_scratchpad

    # 2. Generate Thought
    logging.info(f"Step {step}: Thought 생성 중...")
    prompt_thought = build_prompt(state, stage="thought")
    resp_thought = llm.invoke(prompt_thought)
    thought_text = getattr(resp_thought, "content", None) or str(resp_thought)
    thought = thought_text.strip()
    current_scratchpad += f"\nThought {step}: {thought}"
    state["scratchpad"] = current_scratchpad  # Update state for next prompt

    # Truncate again before next LLM call
    truncated_scratchpad = truncate_scratchpad(current_scratchpad, context_len)
    state["scratchpad"] = truncated_scratchpad

    # 3. Generate Action
    logging.info(f"Step {step}: Action 생성 중...")
    prompt_action = build_prompt(state, stage="action")
    resp_action = llm.invoke(prompt_action)
    action_text = getattr(resp_action, "content", None) or str(resp_action)
    action_type, argument = parse_action(action_text)  # Use the imported parse_action
    current_scratchpad += f"\nAction {step}: {action_type}[{argument}]"
    state["scratchpad"] = current_scratchpad  # Final update for scratchpad



    step = state.get("step", 1)
    allowed = {"Retrieve", "Search", "Lookup", "Finish"}

    # 폴백 로직: 파싱 실패 또는 허용되지 않은 액션
    if (action_type not in allowed) or (
        argument is None or str(argument).strip() == ""
    ):
        logging.warning(
            f"LLM 응답 파싱 실패 또는 허용되지 않은 액션 감지: {action_type}[{argument}]"
        )
        if step <= 1:
            action_type = "Search"
            argument = state.get("question", "")
            thought = (
                thought
                or "Model output did not follow the required format; falling back to Search."
            )
            action_path = action_path or "Start"
            logging.info(f"Step {step}: Search로 폴백: {argument}")
        else:
            # 간단한 best-effort 답: 마지막 패시지 일부를 반환하거나 Unknown
            last_passages = state.get("last_passages", []) or []
            candidate = (
                (
                    last_passages[-1][:300]
                    + ("..." if last_passages and len(last_passages[-1]) > 300 else "")
                )
                if last_passages
                else "Unknown"
            )
            action_type = "Finish"
            argument = candidate
            thought = (
                thought
                or "Model output did not follow the required format; concluding with best-effort answer."
            )
            action_path = action_path or "(fallback)"
            logging.info(f"Step {step}: Finish로 폴백: {argument}")

    # 안전장치: 연속 Search 또는 최대 스텝 근접 시 종료
    max_consec = max_consec_search
    auto_finish_step = auto_finish_step
    prev_consec = int(state.get("consecutive_search", 0) or 0)
    new_consec = prev_consec + 1 if action_type == "Search" else 0

    if (step >= auto_finish_step and action_type != "Finish") or (
        new_consec >= max_consec and action_type == "Search"
    ):
        logging.warning(
            f"안전장치 발동: Step {step} (최대 {auto_finish_step}) 또는 연속 Search {new_consec} (최대 {max_consec})"
        )
        last_passages = state.get("last_passages", []) or []
        candidate = (
            (
                last_passages[-1][:300]
                + ("..." if last_passages and len(last_passages[-1]) > 300 else "")
            )
            if last_passages
            else (state.get("question", "") or "Unknown")
        )
        action_type = "Finish"
        argument = candidate
        thought = (thought or "") + " Reached auto-finish guard."
        action_path = action_path or "(guard)"
        new_consec = 0
        logging.info(f"Step {step}: 안전장치로 Finish: {argument}")

    new_state: Dict = {
        **state,
        "scratchpad": current_scratchpad,
        "action_type": action_type,  # type: ignore
        "argument": argument,
        "consecutive_search": new_consec,
    }
    logging.info(f"node_decide 완료. 다음 액션: {action_type}[{argument}]")
    return new_state


def node_retrieve(state: Dict) -> Dict:
    """Retrieve 도구 실행 → Observation 기록 → 상태 업데이트"""
    step = state.get("step", 1)
    argument = state.get("argument", "") or ""
    logging.info(f"node_retrieve 실행 (Step {step}): {argument}")
    obs = wikipedia_retrieve(argument)
    logging.info(f"node_retrieve 완료. Observation 길이: {len(obs) if obs else 0}")

    new_scratch = state.get("scratchpad", "")
    if new_scratch and not new_scratch.endswith("\n"):
        new_scratch += "\n"
    new_scratch += f"Observation {step}: {obs}"

    last_passages = list(state.get("last_passages", []))
    if obs:
        last_passages.append(obs)

    return {
        **state,
        "scratchpad": new_scratch,
        "last_passages": last_passages,
        "step": step + 1,
    }


def node_search(state: Dict) -> Dict:
    """Search 도구 실행 → Observation 기록 → 상태 업데이트"""
    step = state.get("step", 1)
    argument = state.get("argument", "") or ""
    logging.info(f"node_search 실행 (Step {step}): {argument}")
    obs = web_search(argument)
    logging.info(f"node_search 완료. Observation 길이: {len(obs) if obs else 0}")

    new_scratch = state.get("scratchpad", "")
    if new_scratch and not new_scratch.endswith("\n"):
        new_scratch += "\n"
    new_scratch += f"Observation {step}: {obs}"

    last_passages = list(state.get("last_passages", []))
    if obs:
        last_passages.append(obs)

    return {
        **state,
        "scratchpad": new_scratch,
        "last_passages": last_passages,
        "step": step + 1,
    }


def node_lookup(state: Dict) -> Dict:
    """Lookup 도구 실행 → Observation 기록 → 상태 업데이트"""
    step = state.get("step", 1)
    keyword = state.get("argument", "") or ""
    last_passages = state.get("last_passages", [])
    base_text = last_passages[-1] if last_passages else ""
    logging.info(
        f"node_lookup 실행 (Step {step}): 키워드='{keyword}', 대상 텍스트 길이: {len(base_text)}"
    )
    obs = lookup_keyword(base_text, keyword)
    logging.info(f"node_lookup 완료. Observation 길이: {len(obs) if obs else 0}")

    new_scratch = state.get("scratchpad", "")
    if new_scratch and not new_scratch.endswith("\n"):
        new_scratch += "\n"
    new_scratch += f"Observation {step}: {obs}"

    if obs:
        last_passages = list(last_passages)
        last_passages.append(obs)

    return {
        **state,
        "scratchpad": new_scratch,
        "last_passages": last_passages,
        "step": step + 1,
    }


def node_finish(state: Dict) -> Dict:
    """종료를 표시하고 최종 답변을 저장합니다."""
    step = state.get("step", 1)
    argument = state.get("argument", "") or ""
    logging.info(f"node_finish 실행 (Step {step}): 최종 답변='{argument}'")

    new_scratch = state.get("scratchpad", "")
    if new_scratch and not new_scratch.endswith("\n"):
        new_scratch += "\n"
    new_scratch += f"Observation {step}: Finished."

    return {
        **state,
        "scratchpad": new_scratch,
        "finished": True,
        "answer": argument,
        "step": step + 1,
    }
