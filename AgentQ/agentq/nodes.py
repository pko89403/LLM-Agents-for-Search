"""
에이전트의 핵심 의사결정 및 도구 실행 노드 구현
"""

from typing import Dict, Any
from pydantic import BaseModel, Field
from agentq.state import AgentState, increment_loop_count, add_error, clear_error
from agentq.llm_utils import get_llm_manager
from agentq.prompt_utils import (
    get_prompt_builder, ScratchpadManager,
    extract_action_from_response, extract_critique_decision, clean_response,
    split_output_blocks, extract_commands_and_status, parse_command_line
)
from agentq.tools import get_tool_executor, WebTool
import re
import json

# 릴리즈 토글/가드
ENABLE_CRITIC = True
ENABLE_MCTS_LITE = True

#
# --- Critic 출력 파싱 유틸 ---
class CriticItem(BaseModel):
    cmd: str = Field(..., description="One of the given candidate commands EXACTLY as provided.")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence 0..1")

class CriticList(BaseModel):
    items: list[CriticItem]
def _normalize_cmd(s: str) -> str:
    import re
    s = (s or "").strip()
    s = s.strip('`"\' ')
    s = re.sub(r"\s+", " ", s)
    return s

# Ollama JSON Schema for critic ranking
CRITIC_JSON_SCHEMA = {
    "oneOf": [
        {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["cmd", "score"],
                "properties": {
                    "cmd": {"type": "string", "minLength": 1},
                    "score": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                }
            }
        },
        {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["cmd", "score"],
                        "properties": {
                            "cmd": {"type": "string", "minLength": 1},
                            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                        }
                    }
                }
            },
            "required": ["items"]
        }
    ]
}

def _extract_json_array(text: str):
    import json, re
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        m = re.search(r"\[[\s\S]*\]", text)
        if m:
            return json.loads(m.group(0))
    except Exception:
        return None
    return None


def _parse_critic_output(critic_out: str, candidates: list[str]) -> list[tuple[str, float]]:
    import re
    cand_norm = [_normalize_cmd(c) for c in candidates]
    parsed = _extract_json_array(critic_out)
    print(f"--- CRITIC PARSER: parsed = {parsed} ---")
    items: list[tuple[str, float]] = []

    def clamp01(x):
        try:
            v = float(x)
        except Exception:
            return 0.5
        return 0.0 if v < 0 else 1.0 if v > 1 else v

    # New: handle dict containers and unwrap single-object payloads
    if isinstance(parsed, dict):
        # unwrap common containers or single-object payloads
        if "items" in parsed and isinstance(parsed["items"], list):
            parsed = parsed["items"]
        elif "data" in parsed and isinstance(parsed["data"], list):
            parsed = parsed["data"]
        elif "choices" in parsed and isinstance(parsed["choices"], list):
            parsed = parsed["choices"]
        elif "result" in parsed and isinstance(parsed["result"], list):
            parsed = parsed["result"]
        elif ("cmd" in parsed or "command" in parsed or "action" in parsed or "candidate" in parsed or "text" in parsed):
            parsed = [parsed]

    if isinstance(parsed, list):
        # Fast path A: pure numeric array -> align by index
        try:
            if all(isinstance(x, (int, float)) for x in parsed):
                nums = [clamp01(x) for x in parsed]
                items = []
                for i, c in enumerate(cand_norm):
                    sc = nums[i] if i < len(nums) else 0.5
                    items.append((c, sc))
                # convert back to original text forms
                return [(candidates[i], items[i][1]) for i in range(len(candidates))]
        except Exception:
            pass

        # Fast path B: objects with {"index": int, "score": num}
        try:
            if all(isinstance(x, dict) and ("index" in x) for x in parsed):
                idx_map = {}
                for x in parsed:
                    try:
                        idx = int(x.get("index"))
                        sc = clamp01(x.get("score", 0.5))
                        idx_map[idx] = sc
                    except Exception:
                        continue
                items = []
                for i, c in enumerate(cand_norm):
                    sc = idx_map.get(i, 0.5)
                    items.append((c, sc))
                return [(candidates[i], items[i][1]) for i in range(len(candidates))]
        except Exception:
            pass

        for ent in parsed:
            if isinstance(ent, dict):
                cmd_key = next((k for k in ent.keys() if k.lower() in ("cmd","command","action","candidate","text")), None)
                score_key = next((k for k in ent.keys() if k.lower() in ("score","confidence","prob","p")), None)
                # Use only .get() access for dict fields
                if cmd_key:
                    cmd = _normalize_cmd(str(ent.get(cmd_key, "")))
                    sc = clamp01(ent.get(score_key, 0.5)) if score_key else 0.5
                    if cmd:
                        items.append((cmd, sc))
            elif isinstance(ent, list) and len(ent) >= 1:
                cmd = _normalize_cmd(str(ent[0]))
                sc = clamp01(ent[1]) if len(ent) >= 2 else 0.5
                if cmd:
                    items.append((cmd, sc))
            elif isinstance(ent, str):
                cmd = _normalize_cmd(ent)
                if cmd:
                    items.append((cmd, 0.5))

    scores_map: dict[str, float] = {}
    for i, c in enumerate(candidates):
        cn = cand_norm[i]
        score = None
        for cmd, sc in items:
            n = _normalize_cmd(cmd)
            if n == cn:
                score = sc; break
        if score is None:
            for cmd, sc in items:
                if _normalize_cmd(cmd).casefold() == cn.casefold():
                    score = sc; break
        if score is None:
            for cmd, sc in items:
                n = _normalize_cmd(cmd)
                if n and (n in cn or cn in n):
                    score = sc; break
        if score is None:
            score = 0.5
        scores_map[c] = float(score)
    # If no items matched, set uniform 0.5 for all candidates
    if not items:
        return [(c, 0.5) for c in candidates]
    return [(c, scores_map[c]) for c in candidates]


def _progress_fingerprint(state: AgentState) -> str:
    from hashlib import md5
    url = (state.get("current_url") or "")
    title = (state.get("page_title") or "")
    content = (state.get("page_content") or "")[:1000]
    return md5(f"{url}|{title}|{content}".encode("utf-8")).hexdigest()


async def plan_node(state: AgentState) -> Dict[str, Any]:
    """Plan 노드: 전체 계획 수립"""
    print("📋 Plan 노드 실행 중...")

    try:
        # 프롬프트 구성
        prompt_builder = get_prompt_builder()
        system_prompt = prompt_builder.build_plan_prompt(state)

        # LLM 호출
        llm_manager = get_llm_manager()
        response = await llm_manager.invoke_with_system(
            system_prompt=system_prompt,
            user_message=f"Please create a plan for: {state['objective']}"
        )

        # 응답 정리
        plan = clean_response(response)

        # 상태 업데이트
        state["plan"] = plan

        # 스크래치패드에 추가
        state = ScratchpadManager.add_plan(state, plan)

        print(f"   계획 수립 완료: {plan[:100]}...")
        return {"plan": plan}

    except Exception as e:
        error_msg = f"Plan 노드 실행 중 오류: {str(e)}"
        print(f"❌ {error_msg}")
        state = add_error(state, error_msg)
        return {"plan": "계획 수립에 실패했습니다."}


async def thought_node(state: AgentState) -> Dict[str, Any]:
    """Thought 노드: 후보 생성 → critic 점수화 → 최종 액션 선택"""
    print(f"🤔 Thought 노드 실행 중... (루프 {state['loop_count'] + 1})")

    try:
        # 루프 카운트 증가
        state = increment_loop_count(state)

        # 최신 DOM 스냅샷 확보 (없을 때만)
        if not state.get("page_content"):
            snap = await WebTool.extract_page_content()
            if snap.get("success") and isinstance(snap.get("data"), dict):
                d = snap["data"]
                state["current_url"] = d.get("url")
                state["page_title"] = d.get("title")
                state["page_content"] = (d.get("content") or "")[:500]
                state["observation"] = "페이지 내용 추출 완료"

        prompt_builder = get_prompt_builder()
        llm_manager = get_llm_manager()

        action = None
        thought_text = ""
        cmds = []
        status = "CONTINUE"
        best_cmd = None
        scores = []
        critic_out = "" # Initialize critic_out

        if ENABLE_CRITIC:
            # 후보 커맨드 생성 (복잡한 프롬프트 사용)
            system_prompt = prompt_builder.build_thought_prompt(state)
            resp = await llm_manager.invoke_with_system(
                system_prompt=system_prompt,
                user_message="Propose multiple candidate COMMANDS for the very next step."
            )
            raw = clean_response(resp)
            cmds, status = extract_commands_and_status(raw)
            thought_text = split_output_blocks(raw).get("THOUGHT", "")

            if cmds and ENABLE_MCTS_LITE:
                try:
                    critic_out = ""
                    critic_prompt = prompt_builder.build_critic_prompt(state)
                    # Include strict instruction that EVERY candidate must appear exactly once
                    critic_in = (
                        "You are a ranking critic. For EACH of the following candidate commands, "
                        "assign a confidence score in [0,1]. Include EVERY candidate exactly once; "
                        "the `cmd` field must be a verbatim copy.\n\n"
                        + "\n".join([f"- {c}" for i, c in enumerate(cmds)])
                    )

                    # --- 0) First try: enforce JSON Schema (Ollama structured outputs) ---
                    parsed_scores = None
                    critic_json_raw = None
                    try:
                        critic_json_raw = await llm_manager.invoke_json_schema_with_system(
                            critic_prompt,
                            critic_in,
                            CRITIC_JSON_SCHEMA,
                            model_name=None  # use default model
                        )
                    except Exception as _e0:
                        print(f"⚠️ Critic JSON-schema invoke 오류: {_e0}")

                    if critic_json_raw:
                        try:
                            data = json.loads(critic_json_raw)
                            # Unwrap common containers or single-object payloads
                            if isinstance(data, dict):
                                if "items" in data and isinstance(data["items"], list):
                                    data = data["items"]
                                elif "data" in data and isinstance(data["data"], list):
                                    data = data["data"]
                                elif "choices" in data and isinstance(data["choices"], list):
                                    data = data["choices"]
                                elif "result" in data and isinstance(data["result"], list):
                                    data = data["result"]
                                elif any(k in data for k in ("cmd","command","action","candidate","text")):
                                    data = [data]
                            critic_out = critic_json_raw  # for logging
                            norm_map: dict[str, float] = {}
                            # Case 1: array of objects with cmd/score (ideal)
                            if isinstance(data, list) and all(isinstance(ent, dict) for ent in data):
                                for ent in data:
                                    ncmd = None
                                    sc = 0.5
                                    try:
                                        # tolerate variants, use only .get()
                                        for k in ("cmd", "command", "action", "candidate", "text"):
                                            if k in ent and isinstance(ent.get(k), (str, int, float)):
                                                ncmd = _normalize_cmd(str(ent.get(k)))
                                                break
                                        for k in ("score", "confidence", "prob", "p"):
                                            if k in ent:
                                                sc = float(ent.get(k))
                                                break
                                    except Exception:
                                        pass
                                    if ncmd:
                                        norm_map[ncmd] = max(0.0, min(1.0, sc))
                                if norm_map:
                                    parsed_scores = [(c, float(norm_map.get(_normalize_cmd(c), 0.5))) for c in cmds]
                            # Case 2: array of numbers -> align by index
                            if not parsed_scores and isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
                                nums = [max(0.0, min(1.0, float(x))) for x in data]
                                parsed_scores = [(c, nums[i] if i < len(nums) else 0.5) for i, c in enumerate(cmds)]
                            # Case 3: array of {"index": int, "score": num}
                            if not parsed_scores and isinstance(data, list) and all(isinstance(x, dict) and ("index" in x) for x in data):
                                idx_map = {}
                                for x in data:
                                    try:
                                        idx = int(x.get("index"))
                                        sc = max(0.0, min(1.0, float(x.get("score", 0.5))))
                                        idx_map[idx] = sc
                                    except Exception:
                                        continue
                                parsed_scores = [(c, idx_map.get(i, 0.5)) for i, c in enumerate(cmds)]
                        except Exception as _ejson:
                            print(f"⚠️ Critic JSON-schema 파싱 실패: {_ejson}")
                            parsed_scores = None

                    # --- 1) Try structured output (OpenAI etc.) ---
                    structured = None
                    if not parsed_scores:
                        try:
                            structured = await llm_manager.invoke_structured_with_system(
                                critic_prompt,
                                critic_in,
                                CriticList
                            )
                        except Exception as _e_struct:
                            print(f"⚠️ Critic structured invoke 실패: {_e_struct}")
                            structured = None

                    # Case 1: Pydantic BaseModel 로 반환된 경우만 .items 사용
                    from pydantic import BaseModel as _PBM
                    if (not parsed_scores) and isinstance(structured, _PBM):
                        try:
                            s_items = getattr(structured, "items", None)
                            if s_items:
                                tmp = [(_normalize_cmd(getattr(it, "cmd")), float(getattr(it, "score"))) for it in s_items]
                                norm_map = {cmd: sc for cmd, sc in tmp}
                                parsed_scores = [(c, float(norm_map.get(_normalize_cmd(c), 0.5))) for c in cmds]
                        except Exception as _e:
                            print(f"⚠️ Structured critic 파싱 오류(BaseModel): {_e}")
                            parsed_scores = None

                    # Case 2: dict/list/str 로 반환되면 통합 파서에 위임
                    if (not parsed_scores) and isinstance(structured, (dict, list)):
                        try:
                            import json as _json
                            parsed_scores = _parse_critic_output(_json.dumps(structured), cmds)
                        except Exception as _e:
                            print(f"⚠️ Structured critic 파싱 오류(dict/list): {_e}")
                            parsed_scores = None
                    elif (not parsed_scores) and isinstance(structured, str):
                        try:
                            parsed_scores = _parse_critic_output(structured, cmds)
                        except Exception as _e:
                            print(f"⚠️ Structured critic 파싱 오류(str): {_e}")
                            parsed_scores = None

                    # --- 2) Fallback: unstructured JSON parsing + one repair retry ---
                    if not parsed_scores:
                        critic_out = await llm_manager.invoke_with_system(critic_prompt, critic_in)
                        parsed_scores = _parse_critic_output(critic_out, cmds)
                        need_retry = (not parsed_scores) or all(abs(s - 0.5) < 1e-6 for _, s in parsed_scores)
                        if need_retry:
                            repair_user = (
                                "Your previous output was INVALID. Return ONLY a JSON array where each item is:\n"
                                "{\"cmd\": \"<one of the given candidates EXACTLY>\", \"score\": <float 0..1>}\n"
                                "Do NOT include any explanation or code fences. Include EVERY candidate exactly once.\n\n"
                                "Candidates (copy verbatim):\n" + "\n".join([f"- {c}" for c in cmds])
                            )
                            critic_out2 = await llm_manager.invoke_with_system(critic_prompt, repair_user)
                            try:
                                parsed2 = _parse_critic_output(critic_out2, cmds)
                            except Exception:
                                parsed2 = None
                            if parsed2:
                                parsed_scores = parsed2
                                critic_out = critic_out2
                    try:
                        preview = (critic_out or "<empty>")[:600]
                        print("   Critic 원본 출력(최종):\n" + preview)
                    except Exception:
                        pass

                    if parsed_scores:
                        scores.extend(parsed_scores)
                        try:
                            sample = ", ".join([f"{c}={s:.2f}" for c, s in parsed_scores[:3]])
                            print("   Critic 점수 샘플: " + sample)
                        except Exception:
                            pass

                except Exception as e:
                    print(f"⚠️ Critic/MCTS-lite 블록 오류: {e}")
                    # 안전 폴백: 비워두면 아래 균등 분배 폴백으로 이어집니다.

            # Single consolidated fallback for empty scores and cmds
            if not scores and cmds:
                scores = [(c, 0.5) for c in cmds]

            if scores:
                # Q 통계와 결합 (간단한 UCB-lite)
                qstats = state.get("q_stats") or {}
                alpha = 0.5
                scored = []
                for c, s in scores:
                    q = qstats.get(c, {}).get("Q", 0.0)
                    n = qstats.get(c, {}).get("N", 0)
                    bonus = 0.1 if n == 0 else 0.0
                    total = alpha * s + (1 - alpha) * q + bonus
                    scored.append((total, c))
                scored.sort(reverse=True)
                try:
                    preview = ", ".join([f"{c}~{t:.2f}" for t, c in sorted(scored, reverse=True)[:3]])
                    print(f"   UCB-lite 상위: {preview}")
                except Exception:
                    pass
                best_cmd = scored[0][1]
                action = parse_command_line(best_cmd)
            else:
                # Critic/MCTS-lite 실패 시 Option A (단일 액션)으로 폴백
                print("⚠️ Critic/MCTS-lite 실패 또는 후보 없음. Option A로 폴백합니다.")
                # Fallback to Option A logic
                system_prompt = prompt_builder.build_thought_prompt(state) # 이 프롬프트는 이제 단순화된 버전
                resp = await llm_manager.invoke_with_system(
                    system_prompt=system_prompt,
                    user_message="What should be the next action based on the current situation?"
                )
                thought_text = clean_response(resp)
                action = extract_action_from_response(thought_text) # 단순화된 파서 사용
        else: # ENABLE_CRITIC is False
            # Option A (단일 액션) 프롬프트 사용
            system_prompt = prompt_builder.build_thought_prompt(state) # 이 프롬프트는 이제 단순화된 버전
            resp = await llm_manager.invoke_with_system(
                system_prompt=system_prompt,
                user_message="What should be the next action based on the current situation?"
            )
            thought_text = clean_response(resp)
            action = extract_action_from_response(thought_text) # 단순화된 파서 사용

        # 상태 업데이트
        state["thought"] = thought_text or "다음 행동을 결정했습니다."
        state["candidate_commands"] = cmds
        state["critic_scores"] = [s for _, s in scores] if scores else []
        state["status"] = status
        state["last_command"] = best_cmd if 'best_cmd' in locals() else None
        state["action"] = action

        # 스크래치패드 기록
        state = ScratchpadManager.add_thought(state, state["thought"])
        if action:
            state = ScratchpadManager.add_action(state, action)

        print(f"   사고 과정: {state['thought'][:100]}...")
        if action:
            print(f"   계획된 액션: {action.get('type')}")
        else:
            print("   액션이 추출되지 않았습니다.")

        return {
            "thought": state["thought"],
            "action": action,
            "loop_count": state["loop_count"],
            "candidate_commands": cmds
        }
    except Exception as e:
        error_msg = f"Thought 노드 실행 중 오류: {str(e)}"
        print(f"❌ {error_msg}")
        state = add_error(state, error_msg)
        return {
            "thought": "다음 행동을 결정하는데 실패했습니다.",
            "action": None,
            "loop_count": state["loop_count"]
        }


async def action_node(state: AgentState) -> Dict[str, Any]:
    """Action 노드: 실제 액션 실행"""
    print("⚡ Action 노드 실행 중...")

    try:
        action = state.get("action")

        if not action:
            observation = "실행할 액션이 없습니다."
            print(f"   {observation}")
            state["observation"] = observation
            state = ScratchpadManager.add_observation(state, observation)
            return {"observation": observation}

        # 도구 실행
        tool_executor = get_tool_executor()
        result = await tool_executor.execute_action(action)

        # 관찰 결과 구성
        if result["success"]:
            observation = result["message"]
            if result.get("data"):
                if isinstance(result["data"], dict):
                    # 페이지 정보 업데이트
                    if "url" in result["data"]:
                        state["current_url"] = result["data"]["url"]
                    if "title" in result["data"]:
                        state["page_title"] = result["data"]["title"]
                    if "content" in result["data"]:
                        state["page_content"] = result["data"]["content"][:500]  # 처음 500자만
                        observation += f"\n페이지 내용: {result['data']['content'][:200]}..."
                elif isinstance(result["data"], str):
                    observation += f"\n결과: {result['data'][:200]}..."
        else:
            observation = f"액션 실패: {result['message']}"

        # 상태 업데이트
        state["observation"] = observation
        state = ScratchpadManager.add_observation(state, observation)

        # 에러 상태 초기화 (성공한 경우)
        if result["success"]:
            state = clear_error(state)

        print(f"   실행 결과: {observation[:100]}...")
        return {"observation": observation}

    except Exception as e:
        error_msg = f"Action 노드 실행 중 오류: {str(e)}"
        print(f"❌ {error_msg}")
        observation = f"액션 실행 중 오류가 발생했습니다: {str(e)}"
        state["observation"] = observation
        state = add_error(state, error_msg)
        state = ScratchpadManager.add_observation(state, observation)
        return {"observation": observation}


async def explanation_node(state: AgentState) -> Dict[str, Any]:
    """Explanation 노드: 결과 설명"""
    print("📝 Explanation 노드 실행 중...")

    try:
        # 프롬프트 구성
        prompt_builder = get_prompt_builder()
        system_prompt = prompt_builder.build_explanation_prompt(state)

        # LLM 호출
        llm_manager = get_llm_manager()
        response = await llm_manager.invoke_with_system(
            system_prompt=system_prompt,
            user_message="Please explain what happened and its significance."
        )

        # 응답 정리
        explanation = clean_response(response)

        # 상태 업데이트
        state["explanation"] = explanation
        state = ScratchpadManager.add_explanation(state, explanation)

        print(f"   설명: {explanation[:100]}...")
        return {"explanation": explanation}

    except Exception as e:
        error_msg = f"Explanation 노드 실행 중 오류: {str(e)}"
        print(f"❌ {error_msg}")
        explanation = f"결과 해석 중 오류가 발생했습니다: {str(e)}"
        state["explanation"] = explanation
        state = add_error(state, error_msg)
        return {"explanation": explanation}


async def critique_node(state: AgentState) -> Dict[str, Any]:
    """Critique 노드: 완료 여부 판단"""
    print("🔍 Critique 노드 실행 중...")

    try:
        # 프롬프트 구성
        prompt_builder = get_prompt_builder()
        system_prompt = prompt_builder.build_critique_prompt(state)

        # LLM 호출
        llm_manager = get_llm_manager()
        response = await llm_manager.invoke_with_system(
            system_prompt=system_prompt,
            user_message="Should we continue or is the task complete?"
        )

        # 응답 정리
        critique = clean_response(response)

        # 완료 여부 결정
        done = extract_critique_decision(response)

        # ---- 진행도/루프 가드 ----
        loops = state.get("loop_count", 0)
        min_loops = state.get("min_loops", 3)

        prev_fp = state.get("last_progress_fingerprint")
        curr_fp = _progress_fingerprint(state)
        progress_gain = (prev_fp != curr_fp) and bool(curr_fp)

        if progress_gain:
            state["no_progress_streak"] = 0
            state["last_progress_fingerprint"] = curr_fp
        else:
            state["no_progress_streak"] = state.get("no_progress_streak", 0) + 1

        if loops < min_loops:
            done = False

        if state.get("no_progress_streak", 0) >= 2 and loops >= min_loops:
            done = True
            critique += "\nHeuristic: No progress for multiple steps → stopping."

        # 도메인 휴리스틱 (OpenTable)
        try:
            url = state.get("current_url") or ""
            text = (state.get("page_content") or "").lower()
            if "opentable.com" in url:
                success_keywords = ["reservation confirmed", "complete reservation", "you're all set"]
                if any(k in text for k in success_keywords):
                    done = True
                    critique += "\nHeuristic: OpenTable success indicators found."
        except Exception:
            pass

        # 간단한 Q-통계 업데이트 (세션 내에서만)
        if ENABLE_MCTS_LITE:
            try:
                cmd = state.get("last_command")
                if cmd:
                    stats = state.get("q_stats") or {}
                    ent = stats.get(cmd, {"Q": 0.0, "N": 0})
                    reward = 1.0 if done else (0.2 if progress_gain else 0.0)
                    ent["N"] += 1
                    ent["Q"] = ent["Q"] + (reward - ent["Q"]) / ent["N"]
                    stats[cmd] = ent
                    state["q_stats"] = stats
            except Exception:
                pass

        # 최대 루프 횟수 체크
        if state["loop_count"] >= state["max_loops"]:
            done = True
            critique += f"\n최대 루프 횟수({state['max_loops']})에 도달하여 종료합니다."

        print(f"   루프 {loops}, min_loops {min_loops}, no_progress_streak {state.get('no_progress_streak')}, done={done}")

        # 상태 업데이트
        state["done"] = done
        state = ScratchpadManager.add_critique(state, critique, done)

        status = "완료" if done else "계속"
        print(f"   평가 결과: {status} - {critique[:100]}...")

        return {
            "done": done,
            "critique": critique
        }

    except Exception as e:
        error_msg = f"Critique 노드 실행 중 오류: {str(e)}"
        print(f"❌ {error_msg}")

        # 오류 발생 시 안전하게 종료
        critique = f"평가 중 오류가 발생했습니다: {str(e)}"
        state["done"] = True
        state = add_error(state, error_msg)

        return {
            "done": True,
            "critique": critique
        }


# 라우팅 함수
def should_continue(state: AgentState) -> str:
    """다음 노드 결정"""
    if state.get("done", False):
        return "end"
    else:
        return "thought"


def check_max_loops(state: AgentState) -> str:
    """최대 루프 횟수 체크"""
    if state["loop_count"] >= state["max_loops"]:
        return "critique"  # 강제로 critique로 보내서 종료 처리
    return "action"
