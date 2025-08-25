# -*- coding: utf-8 -*-
"""에이전트의 각 상태(노드)에 해당하는 핵심 로직을 구현합니다."""

import logging
from typing import Any, Dict, List, Optional
import re

from langchain_core.language_models import BaseLanguageModel


from state import LaserState
from tools import ToolKit
from tool_specs import search_items, select_item, next_page, back_to_search, description, features, reviews, buy_now, previous_page
from parsing_utils import parse_observation # parse_observation 함수를 임포트

from prompt_utils import build_prompt, build_scoring_prompt, build_feedback_prompt, build_rethink_prompt, build_manager_prompt # build_prompt, build_scoring_prompt 함수를 임포트

# --- Feature flag: enable a simple micro-agent loop inside Item node ---
ENABLE_ITEM_MICRO_AGENT = True

_PRICE_RE = re.compile(r"\$([\d\.,]+)")
_MAX_PRICE_RE = re.compile(r"price\s*(?:lower than|under)\s*([\d\.]+)", re.IGNORECASE)

def _extract_max_price_from_instruction(instruction: str) -> Optional[float]:
    m = _MAX_PRICE_RE.search(instruction or "")
    try:
        return float(m.group(1)) if m else None
    except Exception:
        return None

def _parse_price_from_obs(obs: str) -> Optional[float]:
    m = _PRICE_RE.search(obs or "")
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except Exception:
        return None

def run_item_micro_agent(state: LaserState, llm: BaseLanguageModel, enable_feedback: bool = False, max_inner_steps: int = 3) -> Dict[str, Any]:
    logging.info("[아이템 마이크로 에이전트] 시작 (원본 LASER 스타일)")
    toolkit = ToolKit(state["_env"])
    obs = state.get("obs", "") or ""
    user_instruction = state.get("user_instruction", "")
    step_count = state.get("step_count", 0)
    raw_action_history = state.get("action_history") or []
    thought_history = state.get("thought_history") or []
    visited = {"description": False, "features": False, "reviews": False}
    additional_info = []
    available_specs = [description, reviews, features, buy_now, previous_page]

    # 리플레이/데모 환경 호환: Item 진입 직후 기대 액션이 Buy Now면 즉시 구매
    def _norm_action(s: Optional[str]) -> str:
        return (s or "").strip().lower()

    # 1) 우선 state.info.expected_action 사용 (Result → Item 전이 직후의 최신값이 더 신뢰도 높음)
    exp_action = (state.get("info") or {}).get("expected_action")

    # 2) 없으면 env의 step_info에서 백업으로 가져오기
    if not exp_action:
        env = state.get("_env")
        try:
            if env and hasattr(env, "get_current_step_info") and callable(env.get_current_step_info):
                step_info = env.get_current_step_info() or {}
                exp_action = step_info.get("expected_action") or step_info.get("expected")
        except Exception as _e:
            logging.debug(f"[아이템 마이크로 에이전트] step_info 조회 실패: {_e}")

    exp_norm = _norm_action(exp_action)
    logging.info(f"[아이템 마이크로 에이전트] 기대 액션 감지: {exp_action}")

    # 3) 휴리스틱: Item 페이지인데 기대 액션이 'click[<item_id>]' 형태로 남아 있는 경우, Buy Now로 보정
    #    (리플레이 로그가 Result 단계의 선택을 그대로 들고 오는 경우가 있음)
    _item_page_invalid_clicks = (
        "click[description]", "click[features]", "click[reviews]",
        "click[< prev]", "click[next >]", "click[back to search]", "click[buy now]"
    )
    if exp_norm.startswith("click[") and exp_norm not in _item_page_invalid_clicks:
        logging.info("[아이템 마이크로 에이전트] 기대 액션이 item_id 클릭 형태여서 Item 단계용으로 보정 → Buy Now")
        exp_action = "click[Buy Now]"
        exp_norm = _norm_action(exp_action)

    if exp_norm == _norm_action("click[Buy Now]"):
        logging.info("[아이템 마이크로 에이전트] 리플레이 기대 액션=Buy Now 감지 → 즉시 구매")
        obs_next, reward, done, info = toolkit.execute({"name": "buy_now", "arguments": {}})
        obs_next = obs_next or ""
        raw_action = info.get("predicted_action", "click[Buy Now]")
        raw_action_history.append(raw_action)
        step_count += 1

        selected_item_id = (
            info.get("selected_item_id")
            or next((a for a in raw_action_history[::-1] if a.startswith("click[") and len(a) > 6), "")
            .split("[")[-1].split("]")[0]
        )
        item_name_match = re.search(r"\n([\w\s\.,\(\)-]+)\nPrice: \$([\d\.,]+(?: to \$[\d\.,]+)?)", obs, re.DOTALL)
        item_name = item_name_match.group(1).strip() if item_name_match else "Unknown Item"
        item_price = item_name_match.group(2).strip() if item_name_match else "N/A"
        selected_item = {
            "item_id": selected_item_id,
            "title": item_name,
            "price": item_price,
            "url": None,
            "source_state": "Item",
        }
        return {
            "obs": obs_next,
            "url": None,
            "last_action": raw_action,
            "step_count": step_count,
            "action_history": raw_action_history,
            "thought_history": thought_history,
            "current_laser_state": "Stopping",
            "route": "to_stop",
            "info": info,
            "selected_item_id": selected_item_id,
            "selected_item": selected_item,
        }

    for _ in range(max_inner_steps):
        # 누적 observation 구성
        full_obs = obs + ("\n" + "\n".join(additional_info) if additional_info else "")
        decision = choose_next_action({**state, "obs": full_obs}, available_specs, llm, enable_feedback)
        llm_action = decision.get("action") or {"name": "previous_page", "arguments": {}}
        llm_action["name"] = (llm_action.get("name") or "").lower()
        action_name = llm_action["name"]
        llm_thought = decision.get("thought", "")
        thought_history.append(llm_thought)

        # Logging: LLM이 선택한 도구와 visited 상태 출력
        logging.info(f"[Item Loop] LLM 선택 도구: {action_name}")
        logging.info(f"[Item Loop] visited 상태: {visited}")

        # Prev는 명시적으로 처리
        if action_name == "previous_page":
            if not all(visited.values()):
                # 탐색 도구 남아있으면 Prev 차단
                fallback = next((k for k, v in visited.items() if not v), "description")
                action_name = fallback
                llm_action = {"name": action_name, "arguments": {}}
                logging.info(f"[Item Loop] Prev 차단 → 대체 도구: {action_name}")
        elif action_name in visited:
            if not visited[action_name]:
                visited[action_name] = True
                logging.info(f"[Item Loop] 처음 열람한 도구: {action_name}")
            elif not all(visited.values()):
                fallback = next((k for k, v in visited.items() if not v), "description")
                action_name = fallback
                llm_action = {"name": action_name, "arguments": {}}
                logging.info(f"[Item Loop] 중복 도구 선택 → 대체 도구: {action_name}")

        # 툴 실행
        obs_next, reward, done, info = toolkit.execute(llm_action)
        obs_next = obs_next or ""
        raw_action = info.get("predicted_action", f"{action_name}()")
        raw_action_history.append(raw_action)
        step_count += 1

        # Buy Now일 경우 즉시 종료
        if action_name == "buy_now":
            logging.info("[아이템 마이크로 에이전트] Buy Now 선택됨 → 종료")
            selected_item_id = (
                info.get("selected_item_id")
                or next((a for a in raw_action_history[::-1] if a.startswith("click[") and len(a) > 6), "")
                .split("[")[-1].split("]")[0]
            )
            item_name_match = re.search(r"\n([\w\s\.,\(\)-]+)\nPrice: \$([\d\.,]+(?: to \$[\d\.,]+)?)", obs, re.DOTALL)
            item_name = item_name_match.group(1).strip() if item_name_match else "Unknown Item"
            item_price = item_name_match.group(2).strip() if item_name_match else "N/A"
            selected_item = {
                "item_id": selected_item_id,
                "title": item_name,
                "price": item_price,
                "url": None,
                "source_state": "Item",
            }
            return {
                "obs": obs_next,
                "url": None,
                "last_action": raw_action,
                "step_count": step_count,
                "action_history": raw_action_history,
                "thought_history": thought_history,
                "current_laser_state": "Stopping",
                "route": "to_stop",
                "info": info,
                "selected_item_id": selected_item_id,
                "selected_item": selected_item,
            }

        # 종료 조건: 에러 or done
        if done or info.get("error"):
            logging.info(f"[아이템 마이크로 에이전트] done=True or 에러 발생 → 종료 (action={action_name}, error={info.get('error')})")
            return {
                "obs": obs_next,
                "url": None,
                "last_action": raw_action,
                "step_count": step_count,
                "action_history": raw_action_history,
                "thought_history": thought_history,
                "current_laser_state": "Stopping",
                "route": "to_stop",
                "info": info,
            }

        # 정보 도구 클릭 시 추가 정보 저장
        if action_name in visited:
            block = f"{action_name}:\n{obs_next.strip()}\n"
            additional_info.append(block)

        # 다음 루프용 obs 갱신
        obs = obs_next

    # 최대 스텝 도달 → Prev로 종료
    logging.info("[아이템 마이크로 에이전트] 탐색 종료 → Prev")
    obs_next, reward, done, info = toolkit.execute({"name": "previous_page", "arguments": {}})
    obs_next = obs_next or ""
    raw_action = info.get("predicted_action", "") or "click[< Prev]"
    return {
        "obs": obs_next,
        "url": None,
        "last_action": raw_action,
        "step_count": step_count + 1,
        "action_history": raw_action_history + [raw_action],
        "thought_history": thought_history,
        "current_laser_state": "Result",
        "route": "to_result",
        "info": info,
    }


def add_or_update_buffer(state: LaserState, candidate: dict) -> None:
    """메모리 버퍼에 후보 아이템을 추가하거나 업데이트합니다."""
    logging.info(f"[Memory Buffer] add_or_update_buffer 호출됨. 후보: {candidate.get('item_id')}, 현재 버퍼 크기: {len(state.get('memory_buffer', []))}")
    buf = state.get("memory_buffer", [])
    # item_id를 기준으로 기존 후보를 찾습니다.
    idx = next((i for i, c in enumerate(buf) if c.get("item_id") == candidate.get("item_id")), None)

    # step_count와 times_seen 업데이트
    candidate["last_seen_step"] = state.get("step_count", 0)
    candidate["times_seen"] = (buf[idx].get("times_seen", 0) + 1) if idx is not None else 1

    if idx is None:
        # 새로운 후보이면 추가
        buf.append(candidate)
        logging.info(f"[Memory Buffer] 새 아이템 추가됨: {candidate.get('item_id')}")
    else:
        # 기존 후보이면 병합 업데이트
        # 기존 값 중 None이나 빈 문자열이 아닌 값만 새 값으로 덮어씁니다.
        merged = {**buf[idx], **{k: v for k, v in candidate.items() if v not in (None, "")}}

        # actions_taken 병합 (중복 제거)
        at = list(dict.fromkeys((buf[idx].get("actions_taken") or []) + (candidate.get("actions_taken") or [])))
        merged["actions_taken"] = at

        buf[idx] = merged
        logging.info(f"[Memory Buffer] 기존 아이템 업데이트됨: {candidate.get('item_id')}")

    state["memory_buffer"] = buf
    logging.info(f"[Memory Buffer] 작업 후 버퍼 크기: {len(state.get('memory_buffer', []))}")


def score_item_with_llm(item_info: Dict[str, Any], user_instruction: str, llm: BaseLanguageModel) -> float:
    """LLM을 사용하여 아이템이 사용자 지시사항에 얼마나 잘 맞는지 점수를 매깁니다."""
    logging.info("--- LLM 기반 아이템 스코어링 시작 ---")

    # 프롬프트 구성 (prompt_utils.build_scoring_prompt 사용)
    messages = build_scoring_prompt(item_info, user_instruction)

    try:
        response = llm.invoke(messages)
        logging.info(f"스코어링 LLM 응답: {response.content}")

        # LLM 응답에서 JSON 파싱 (단일 중괄호 사용)
        score_match = re.search(r"{\s*\"score\"\s*:\s*([\d\.]+)\s*}", response.content)
        if score_match:
            score = float(score_match.group(1))
            return max(0.0, min(1.0, score)) # 0.0 ~ 1.0 범위로 제한
        else:
            logging.warning("LLM 응답에서 점수를 파싱할 수 없습니다. 기본 점수 0.5를 반환합니다.")
            return 0.5
    except Exception as e:
        logging.error(f"스코어링 LLM 호출 중 오류 발생: {e}")
        return 0.0 # 오류 발생 시 낮은 점수 반환


def should_rethink_based_on_feedback(feedback: str) -> bool:
    """피드백을 분석하여 재고가 필요한지 판단합니다."""
    if not feedback:
        return False

    feedback_lower = feedback.lower()

    # 강한 부정적 키워드들
    strong_negative_keywords = [
        "wrong", "mistake", "incorrect", "should not", "error", "bad choice",
        "inappropriate", "not suitable", "doesn't match", "not relevant",
        "poor decision", "reconsider", "think again", "not right"
    ]

    # 약한 부정적 키워드들 (더 많은 컨텍스트가 필요)
    weak_negative_keywords = [
        "but", "however", "although", "consider", "might want to", "perhaps",
        "could be better", "alternative", "instead"
    ]

    # 긍정적 키워드들 (재고 불필요)
    positive_keywords = [
        "good", "correct", "right", "appropriate", "suitable", "matches",
        "relevant", "well done", "excellent", "perfect", "accurate"
    ]

    # 강한 부정적 키워드가 있으면 재고 필요
    if any(keyword in feedback_lower for keyword in strong_negative_keywords):
        logging.info(f"강한 부정적 피드백 감지: {feedback[:100]}...")
        return True

    # 긍정적 키워드가 있으면 재고 불필요
    if any(keyword in feedback_lower for keyword in positive_keywords):
        logging.info(f"긍정적 피드백 감지: {feedback[:100]}...")
        return False

    # 약한 부정적 키워드가 있고 긍정적 키워드가 없으면 재고 고려
    if any(keyword in feedback_lower for keyword in weak_negative_keywords):
        # 문장의 길이와 구조를 고려하여 판단
        if len(feedback.split()) > 5:  # 5단어 이상의 피드백일 경우 더 신중하게 판단
            logging.info(f"약한 부정적 피드백 감지 (긴 피드백): {feedback[:100]}...")
            return True
        else:
            logging.info(f"약한 부정적 피드백 감지 (짧은 피드백): {feedback[:100]}...")
            return False

    # 기본적으로 재고하지 않음
    logging.info(f"중립적 피드백: {feedback[:100]}...")
    return False


def get_feedback_from_manager(state: LaserState, observation: str, rationale: str, action: str, llm: BaseLanguageModel) -> str:
    """매니저로부터 피드백을 받습니다."""
    logging.info("--- 매니저 피드백 요청 시작 ---")

    try:
        # 히스토리 구성
        action_history = state.get("action_history", [])
        thought_history = state.get("thought_history", [])

        history_str = ""
        for i in range(max(len(action_history), len(thought_history))):
            if i < len(thought_history):
                history_str += f"Rationale{i}: {thought_history[i]}\n"
            if i < len(action_history):
                history_str += f"Action{i}: {action_history[i]}\n"

        messages = build_manager_prompt(history_str, observation, rationale, action)
        response = llm.invoke(messages)
        feedback = response.content.strip()
        logging.info(f"매니저 피드백: {feedback}")
        return feedback
    except Exception as e:
        logging.error(f"매니저 피드백 요청 중 오류 발생: {e}")
        return "피드백을 받을 수 없습니다."


def rethink_action_with_feedback(state: LaserState, tool_specs: List[Dict], llm: BaseLanguageModel,
                                original_rationale: str, original_action: Dict, feedback: str) -> Dict:
    """피드백을 바탕으로 행동을 재고합니다."""
    logging.info("--- 피드백 기반 행동 재고 시작 ---")

    # 재고 히스토리 확인 (무한 루프 방지)
    rethink_history = state.get("rethink_history", [])
    current_step = state.get("step_count", 0)

    # 같은 스텝에서 이미 재고했는지 확인
    step_rethinks = [r for r in rethink_history if r.get("step") == current_step]
    if len(step_rethinks) >= 2:  # 한 스텝에서 최대 2번까지만 재고 허용
        logging.warning(f"스텝 {current_step}에서 이미 {len(step_rethinks)}번 재고했습니다. 원래 행동을 유지합니다.")
        return {
            "action": original_action,
            "thought": original_rationale + "\n(Rethink limit reached, keeping original action.)",
            "feedback": feedback
        }

    try:
        observation = state.get("obs", "")
        action_str = f"{original_action.get('name', 'Unknown')}({original_action.get('arguments', {})})"

        # 이전 재고 히스토리를 포함한 컨텍스트 구성
        rethink_context = ""
        if step_rethinks:
            rethink_context = "\n\nPrevious rethink attempts in this step:\n"
            for i, prev_rethink in enumerate(step_rethinks):
                rethink_context += f"Attempt {i+1}: {prev_rethink.get('original_action')} -> {prev_rethink.get('rethought_action')}\n"
                rethink_context += f"Feedback: {prev_rethink.get('feedback')}\n"

        # 더미 LLM 처리
        if getattr(llm, "is_dummy", False):
            # 더미 LLM의 경우 간단한 응답 시뮬레이션
            rethink_action = {"name": "back_to_search", "arguments": {}}
            rethink_thought = "(dummy) rethinking based on feedback"
            logging.info(f"더미 LLM 재고 결정: {rethink_action}")
        else:
            llm_with_tools = llm.bind_tools(tool_specs)
            messages = build_rethink_prompt(observation, original_rationale, action_str, feedback + rethink_context, tool_specs)
            response = llm_with_tools.invoke(messages)

            rethink_thought = response.content or ""
            rethink_action = None

            if response.tool_calls:
                tool_call = response.tool_calls[0]
                rethink_action = {"name": tool_call.get("name"), "arguments": tool_call.get("args", {})}
                logging.info(f"재고된 행동: {rethink_action}")
            else:
                # 재고에서도 툴 호출이 없으면 원래 행동 유지
                rethink_action = original_action
                rethink_thought += "\n(Rethink failed, keeping original action.)"

        # 재고 히스토리에 기록 (더미/실제 LLM 모두)
        if rethink_action and rethink_action != original_action:
            rethink_record = {
                "step": current_step,
                "original_action": action_str,
                "rethought_action": f"{rethink_action.get('name')}({rethink_action.get('arguments', {})})",
                "feedback": feedback,
                "timestamp": current_step
            }
            rethink_history.append(rethink_record)
            state["rethink_history"] = rethink_history

        # 재고 결과에 메타데이터 추가
        rethink_thought += f"\n(Rethink attempt {len(step_rethinks) + 1} for step {current_step})"

        return {"action": rethink_action, "thought": rethink_thought, "feedback": feedback}

    except Exception as e:
        logging.error(f"행동 재고 중 오류 발생: {e}")
        return {"action": original_action, "thought": original_rationale + f"\n(Rethink failed: {e})", "feedback": feedback}


def choose_next_action(state: LaserState, tool_specs: List[Dict], llm: BaseLanguageModel, enable_feedback: bool = False) -> Dict:
    """LLM을 호출하여 다음 행동을 결정하는 핵심 함수입니다.

    Args:
        state: 현재 에이전트의 상태.
        tool_specs: 현재 상태에서 LLM이 사용할 수 있는 도구 명세 리스트.
        llm: 사용할 언어 모델 인스턴스.
        enable_feedback: 피드백 시스템을 활성화할지 여부.

    Returns:
        LLM이 결정한 행동 (딕셔너리 형태: {"action": {"name": "tool_name", "arguments": {}}, "thought": "LLM의 생각", "feedback": "매니저 피드백"}).
    """
    logging.info("--- LLM 호출 시작 ---")
    current_laser_state = state.get("current_laser_state")
    parsed_obs = parse_observation(state.get("obs", ""))

    # 1. 기본 프롬프트로 LLM 호출
    try:
        # Dummy LLM 모드 처리
        if getattr(llm, "is_dummy", False):
            # ... (기존 dummy LLM 로직은 변경 없음) ...
            step_info = state.get("_env").get_current_step_info() if state.get("_env") else None
            if step_info:
                action_str = step_info.get('action_executed_in_env')
                name = "back_to_search"
                args = {}
                if action_str.startswith("search["):
                    name = "search"
                    args = {"keywords": action_str[len("search["):-1]}
                elif action_str.startswith("click["):
                    inside = action_str[len("click["):-1]
                    key = inside.strip()
                    if key in ("description", "features", "reviews"):
                        name = key
                    elif key == "Buy Now":
                        name = "buy_now"
                    elif key == "< Prev":
                        name = "prev"
                    elif key == "Next >":
                        name = "next"
                    elif key == "Back to Search":
                        name = "back_to_search"
                    else:
                        name = "select_item"
                        args = {"item_id": key}
                llm_action = {"name": name, "arguments": args}
                llm_thought = "(dummy) following recorded action"
                logging.info(f"Dummy LLM 결정 (툴): {llm_action}")
                return {"action": llm_action, "thought": llm_thought}

        # 실제 LLM 호출
        llm_with_tools = llm.bind_tools(tool_specs)
        messages = build_prompt(state, parsed_obs, current_laser_state, prompt_type="default")
        response = llm_with_tools.invoke(messages)
        logging.info(f"LLM 응답: {response}")

        llm_thought = response.content or ""
        llm_action = None

        if response.tool_calls:
            tool_call = response.tool_calls[0]
            llm_action = {"name": tool_call.get("name"), "arguments": tool_call.get("args", {})}
            logging.info(f"LLM 결정 (툴): {llm_action}")

        # 2. LLM이 툴 호출을 반환하지 않은 경우, 자가 교정 시도
        if not llm_action:
            logging.warning("LLM이 유효한 툴을 호출하지 않았습니다. 자가 교정을 시도합니다.")

            # 매핑 프롬프트로 재호출
            mapping_messages = build_prompt(state, parsed_obs, current_laser_state, prompt_type="mapping", rationale=llm_thought)
            # tool_choice를 사용하여 툴 호출을 강제합니다.
            correction_response = llm_with_tools.invoke(mapping_messages, tool_choice="any")
            logging.info(f"자가 교정 LLM 응답: {correction_response}")

            if correction_response.tool_calls:
                tool_call = correction_response.tool_calls[0]
                llm_action = {"name": tool_call.get("name"), "arguments": tool_call.get("args", {})}
                llm_thought += "\n(Self-correction: Mapped rationale to a valid action.)"
                logging.info(f"자가 교정 성공 (툴): {llm_action}")
            else:
                # 자가 교정도 실패한 경우, 안전한 기본 액션으로 폴백
                logging.error("자가 교정 실패. 안전한 기본 액션으로 폴백합니다.")
                llm_action = {"name": "back_to_search", "arguments": {}}
                llm_thought += "\n(Self-correction failed. Falling back to default action.)"

        # 3. 피드백 시스템 적용 (활성화된 경우)
        feedback = None
        if enable_feedback and llm_action:
            observation = state.get("obs", "")
            action_str = f"{llm_action.get('name', 'Unknown')}({llm_action.get('arguments', {})})"

            # 매니저로부터 피드백 받기
            feedback = get_feedback_from_manager(state, observation, llm_thought, action_str, llm)

            # 피드백이 부정적이면 재고 시도
            if should_rethink_based_on_feedback(feedback):
                logging.info("부정적 피드백 감지. 행동을 재고합니다.")
                rethink_result = rethink_action_with_feedback(state, tool_specs, llm, llm_thought, llm_action, feedback)
                return rethink_result
            else:
                logging.info("긍정적 피드백 또는 중립적 피드백. 원래 행동을 유지합니다.")

        return {"action": llm_action, "thought": llm_thought, "feedback": feedback}

    except Exception as e:
        logging.error(f"LLM 호출 중 심각한 오류 발생: {e}")
        return {"action": {"name": "back_to_search", "arguments": {}}, "thought": f"LLM call failed: {e}", "feedback": None}



# 노드 함수들은 이제 llm 인자를 받습니다。

def node_search_space(state: LaserState, llm: BaseLanguageModel, enable_feedback: bool = False) -> Dict[str, Any]:
    """'Search' 상태 공간 노드: 사용자 지시를 바탕으로 검색어를 생성하고 검색을 실행합니다."""
    logging.info("\n[노드] Search 상태 공간 진입")

    # 1. LLM을 호출하여 다음 행동 결정
    allowed_tool_specs = [search_items]
    llm_decision = choose_next_action(state, allowed_tool_specs, llm, enable_feedback)
    llm_action = llm_decision["action"]
    llm_thought = llm_decision["thought"]
    feedback = llm_decision.get("feedback")

    # 2. 결정된 행동을 ToolKit을 통해 실행
    toolkit = ToolKit(state["_env"])
    obs, reward, done, info = toolkit.execute(llm_action)

    # 환경이 반환한 관찰은 None일 수 있으므로 방어
    obs = obs or ""

    # 로깅/디버깅: 환경에 실제로 보낸 액션 문자열을 그대로 사용
    raw_action_str = (
        info.get("predicted_action")
        if "predicted_action" in info
        else f"search[{llm_action.get('arguments', {}).get('keywords', '')}]"
    )
    logging.info(f"  - 실행된 액션: {raw_action_str}")

    # 마지막 스텝 도달 또는 에러 시 즉시 정지
    if done or info.get("error"):
        return {
            "obs": obs,
            "url": None,
            "last_action": raw_action_str,
            "step_count": state.get("step_count", 0) + 1,
            "action_history": (state.get("action_history") or []) + [raw_action_str],
            "thought_history": (state.get("thought_history") or []) + [llm_thought],
            "current_laser_state": "Stopping",
            "route": "to_stop",
            "info": info,
        }

    # 3. 상태 업데이트
    new_state = {
        "obs": obs,
        "url": None,
        "last_action": raw_action_str,
        "step_count": state.get("step_count", 0) + 1,
        "action_history": (state.get("action_history") or []) + [raw_action_str],
        "thought_history": (state.get("thought_history") or []) + [llm_thought],
        "current_laser_state": "Result",
        "route": "to_result",
        "info": info,
    }

    # 피드백이 있으면 저장
    if feedback:
        feedback_history = state.get("feedback_history", [])
        feedback_history.append({
            "step": state.get("step_count", 0) + 1,
            "state": "Search",
            "feedback": feedback,
            "action": raw_action_str,
            "thought": llm_thought
        })
        new_state["feedback_history"] = feedback_history

    return new_state


def node_result_space(state: LaserState, llm: BaseLanguageModel, enable_feedback: bool = False) -> Dict[str, Any]:
    """'Result' 상태 공간 노드: 검색 결과 목록에서 다음 행동을 결정합니다."""
    logging.info("\n[노드] Result 상태 공간 진입")

    # 1. LLM을 호출하여 다음 행동 결정
    allowed_tool_specs = [select_item, next_page, back_to_search]
    llm_decision = choose_next_action(state, allowed_tool_specs, llm, enable_feedback)
    llm_action = llm_decision["action"]
    llm_thought = llm_decision["thought"]
    feedback = llm_decision.get("feedback")

    llm_action["name"] = (llm_action.get("name") or "").lower()

    action_name_lower = llm_action["name"]

    # 2. 결정된 행동을 ToolKit을 통해 실행
    toolkit = ToolKit(state["_env"])
    obs, reward, done, info = toolkit.execute(llm_action)
    obs = obs or "" # 관찰이 None일 경우 방어

    # 로깅: 환경이 실제로 이해한 액션 문자열을 사용
    raw_action_str = info.get("predicted_action", "")
    logging.info(f"  - 실행된 액션: {raw_action_str}")

    # 마지막 스텝이거나 에러 발생 시 즉시 종료
    if done or info.get("error"):
        return {
            "obs": obs,
            "url": None,
            "last_action": raw_action_str,
            "step_count": state.get("step_count", 0) + 1,
            "action_history": (state.get("action_history") or []) + [raw_action_str],
            "thought_history": (state.get("thought_history") or []) + [llm_thought],
            "current_laser_state": "Stopping",
            "route": "to_stop",
            "info": info,
        }

    # 3. 상태 업데이트 및 다음 라우트 결정
    next_laser_state = "Result"  # 기본값은 Result 상태 유지 (예: 다음 페이지)
    route = "stay_result"

    if action_name_lower == "select_item":
        next_laser_state = "Item"
        route = "to_item"
        # memory_buffer에 아이템 추가/업데이트
        selected_item_id = llm_action.get("arguments", {}).get("item_id")
        if selected_item_id:
            # parsed_obs에서 해당 아이템 정보 찾기
            current_parsed_obs = parse_observation(state["obs"])
            item_info = next((item for item in current_parsed_obs.get("items", []) if item.get("item_id") == selected_item_id), None)
            if item_info:
                candidate_item = {
                    "item_id": item_info.get("item_id"),
                    "title": item_info.get("name"),
                    "price": item_info.get("price_str"),
                    "url": None,
                    "page": current_parsed_obs.get("page_info", {}).get("current_page"),
                    "keywords": state.get("user_instruction", "").split(),
                    "snapshot_excerpt": state.get("obs", "")[:500],
                    "rationale": llm_thought,
                    "source_state": "Result",
                    "actions_taken": [raw_action_str],
                }
                # LLM 스코어링
                score = score_item_with_llm(candidate_item, state.get("user_instruction", ""), llm)
                candidate_item["score"] = score
                logging.info(f"[Memory Buffer] candidate_item 준비됨: {candidate_item.get('item_id')}, 점수: {score}")
                add_or_update_buffer(state, candidate_item)

    elif action_name_lower == "back_to_search":
        next_laser_state = "Search"
        route = "to_search"
    # next_page는 Result 상태 유지

    new_state = {
        "obs": obs,
        "url": None,
        "last_action": raw_action_str,
        "step_count": state.get("step_count", 0) + 1,
        "action_history": (state.get("action_history") or []) + [raw_action_str],
        "thought_history": (state.get("thought_history") or []) + [llm_thought],
        "current_laser_state": next_laser_state,
        "route": route,
        "info": info,
    }

    # 피드백이 있으면 저장
    if feedback:
        feedback_history = state.get("feedback_history", [])
        feedback_history.append({
            "step": state.get("step_count", 0) + 1,
            "state": "Result",
            "feedback": feedback,
            "action": raw_action_str,
            "thought": llm_thought
        })
        new_state["feedback_history"] = feedback_history

    return new_state


def node_item_space(state: LaserState, llm: BaseLanguageModel, enable_feedback: bool = False) -> Dict[str, Any]:
    """'Item' 상태 공간 노드: 상품 상세 페이지에서 다음 행동을 결정합니다."""
    logging.info("\n[노드] Item 상태 공간 진입")

    # If enabled, use the small internal loop to better match original behavior
    allowed_tool_specs = [description, features, reviews, buy_now, previous_page]
    llm_decision = choose_next_action(state, allowed_tool_specs, llm, enable_feedback)
    llm_action = llm_decision["action"]
    llm_thought = llm_decision["thought"]
    feedback = llm_decision.get("feedback")

    # If enabled, use the small internal loop to better match original behavior
    if ENABLE_ITEM_MICRO_AGENT:
        return run_item_micro_agent(state, llm, enable_feedback)

    # 2. 결정된 행동을 ToolKit을 통해 실행
    llm_action["name"] = (llm_action.get("name") or "").lower()
    canonical_name = llm_action["name"]

    toolkit = ToolKit(state["_env"])
    obs, reward, done, info = toolkit.execute({"name": canonical_name, "arguments": llm_action.get("arguments", {})})
    obs = obs or "" # 관찰이 None일 경우 방어

    # 로깅: 환경이 실제로 이해한 액션 문자열을 사용
    raw_action_str = info.get("predicted_action", "")
    logging.info(f"  - 실행된 액션: {raw_action_str}")

    # 마지막 스텝이거나 에러 발생 시 즉시 종료
    if done or info.get("error"):
        return {
            "obs": obs,
            "url": None,
            "last_action": raw_action_str,
            "step_count": state.get("step_count", 0) + 1,
            "action_history": (state.get("action_history") or []) + [raw_action_str],
            "thought_history": (state.get("thought_history") or []) + [llm_thought],
            "current_laser_state": "Stopping",
            "route": "to_stop",
            "info": info,
        }

    # 3. 상태 업데이트 및 다음 라우트 결정
    next_laser_state = "Item"  # 기본값은 Item 상태 유지
    route = "stay_item"

    if action_name_lower == "buy_now":
        next_laser_state = "Stopping"
        route = "to_stop"
        # (이하 로직은 buy_now에 대한 아이템 정보 수집으로, 기존 로직 유지)
        current_item_info = parse_observation(state["obs"])
        item_name_match = re.search(r"\n([\w\s\.,\(\)-]+)\nPrice: \$([\d\.,]+(?: to \$[\d\.,]+)?)", state["obs"], re.DOTALL)
        item_name = item_name_match.group(1).strip() if item_name_match else "Unknown Item"
        item_price = item_name_match.group(2).strip() if item_name_match else "N/A"
        chosen_id = (info.get("selected_item_id") if isinstance(info, dict) else None) or state.get("last_action", "").split("[")[-1].split("]")[0]
        candidate_item = {
            "item_id": chosen_id,
            "title": item_name,
            "price": item_price,
            "url": None,
            "keywords": state.get("user_instruction", "").split(),
            "snapshot_excerpt": state.get("obs", "")[:500],
            "rationale": llm_thought,
            "source_state": "Item",
            "actions_taken": (state.get("actions_taken", []) or []) + [raw_action_str],
            "features_summary": current_item_info.get("item_details_text", "") if current_item_info.get("features_viewed") else "",
            "reviews_summary": current_item_info.get("item_details_text", "") if current_item_info.get("reviews_viewed") else "",
        }
        score = score_item_with_llm(candidate_item, state.get("user_instruction", ""), llm)
        candidate_item["score"] = score
        add_or_update_buffer(state, candidate_item)
        state["selected_item"] = candidate_item

    elif action_name_lower in ("previous_page", "prev", "previous"):
        next_laser_state = "Result"
        route = "to_result"
        # (이하 로직은 prev 선택 시 아이템 정보 업데이트로, 기존 로직 유지)
        current_item_id = (info.get("selected_item_id") if isinstance(info, dict) else None) or state.get("last_action", "").split("[")[-1].split("]")[0]
        if current_item_id:
            current_item_info = parse_observation(state["obs"])
            item_name_match = re.search(r"\n([\w\s\.,\(\)-]+)\nPrice: \$([\d\.,]+(?: to \$[\d\.,]+)?)", state["obs"], re.DOTALL)
            item_name = item_name_match.group(1).strip() if item_name_match else "Unknown Item"
            item_price = item_name_match.group(2).strip() if item_name_match else "N/A"
            candidate_item = {
                "item_id": current_item_id,
                "title": item_name,
                "price": item_price,
                "url": None,
                "keywords": state.get("user_instruction", "").split(),
                "snapshot_excerpt": state.get("obs", "")[:500],
                "rationale": llm_thought,
                "source_state": "Item",
                "actions_taken": (state.get("actions_taken", []) or []) + [raw_action_str],
                "features_summary": current_item_info.get("item_details_text", "") if current_item_info.get("features_viewed") else "",
                "reviews_summary": current_item_info.get("item_details_text", "") if current_item_info.get("reviews_viewed") else "",
            }
            add_or_update_buffer(state, candidate_item)

    # description, features, reviews는 Item 상태 유지

    new_state = {
        "obs": obs,
        "url": None,
        "last_action": raw_action_str,
        "step_count": state.get("step_count", 0) + 1,
        "action_history": (state.get("action_history") or []) + [raw_action_str],
        "thought_history": (state.get("thought_history") or []) + [llm_thought],
        "current_laser_state": next_laser_state,
        "route": route,
        "info": info,
    }

    # 피드백이 있으면 저장
    if feedback:
        feedback_history = state.get("feedback_history", [])
        feedback_history.append({
            "step": state.get("step_count", 0) + 1,
            "state": "Item",
            "feedback": feedback,
            "action": raw_action_str,
            "thought": llm_thought
        })
        new_state["feedback_history"] = feedback_history

    return new_state



def node_stopping_space(state: LaserState) -> Dict[str, Any]:
    """'Stopping' 상태 공간 노드: 최종 결과를 정리하고 종료합니다."""
    logging.info("\n[노드] Stopping 상태 공간 진입")

    selected_item = state.get("selected_item")

    if selected_item:
        logging.info("  - 에이전트가 최종 아이템을 성공적으로 선택했습니다.")
        return {"selected_item": selected_item}
    else:
        logging.warning("  - 최종 아이템이 선택되지 않았습니다. 메모리 버퍼에서 백업 전략을 실행합니다.")
        mem_buffer = state.get("memory_buffer", [])

        if not mem_buffer:
            logging.warning("  - 메모리 버퍼가 비어 있습니다. 선택할 아이템이 없습니다.")
            return {"selected_item": {"note": "최종 선택된 아이템 없음 (메모리 버퍼 비어있음)"}}

        # 백업 전략:
        # 1. score가 있는 경우 최고 점수 선택
        # 2. score가 없거나 동일한 경우 last_seen_step (최신) 우선
        # 3. last_seen_step도 동일한 경우 times_seen (자주 본) 우선

        # 정렬을 위한 키 함수
        def sort_key(item):
            score = item.get("score", -1.0) # score가 없으면 낮은 값으로
            last_seen = item.get("last_seen_step", -1) # 없으면 낮은 값으로
            times_seen = item.get("times_seen", -1) # 없으면 낮은 값으로
            # score는 내림차순, last_seen_step 내림차순, times_seen 내림차순
            return (score, last_seen, times_seen)

        # 버퍼를 정렬하여 가장 좋은 후보를 찾습니다。
        # reverse=True로 내림차순 정렬 (높은 점수, 최신 스텝, 많이 본 순)
        sorted_candidates = sorted(mem_buffer, key=sort_key, reverse=True)

        best_candidate = sorted_candidates[0]
        logging.info(f"  - 백업 전략으로 아이템 선택: {best_candidate.get('title', '제목 없음')} (ID: {best_candidate.get('item_id', 'N/A')})")

        return {"selected_item": best_candidate}
