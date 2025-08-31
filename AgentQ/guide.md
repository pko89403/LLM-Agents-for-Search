좋습니다. 파일 구조는 그대로 유지하면서, 논문(Agent Q)에서 요구하는 출력 포맷(PLAN/THOUGHT/COMMAND/STATUS), 행동 문법(CLICK/GOTO/TYPE/SUBMIT/CLEAR/SCROLL/ASK USER HELP), DOM 중간 표현(ID 태깅), 자기평가(critic) 기반 후보 랭킹 + MCTS-lite 선택기를 DPO/강화학습 없이 넣어 동작하도록 코드를 수정했습니다.
아래 패치를 그대로 적용하면 됩니다. (논문에서 정의한 출력/행동 규격과 OpenTable 실험 셋업에 맞춘 구현입니다.  ￼  ￼)

참고: 자동 편집 도구 연결 이슈로 저장소에 바로 적용하지 못해, 적용 가능한 멀티‑파일 패치 지시문을 제공합니다. 각 변경은 “어디를 바꾸고 → 무엇을 넣는지”가 명시되어 있어 그대로 적용하면 됩니다.

⸻

1) 변경 사항 요약
	•	프롬프트/출력 표준화: LLM이 다음 형식으로만 응답하도록 강제
    ```
    PLAN:
    THOUGHT:
    COMMANDS:
    - <후보1>
    - <후보2>
    STATUS:
    CONTINUE|COMPLETE
    ```
    (논문 Figure 2의 입력/출력 규격 반영.  ￼)

	•	행동 문법 확장:
GOTO[URL=…], CLICK[ID=…], TYPE[ID=…][TEXT=…], SUBMIT[ID=…], CLEAR[ID=…], SCROLL[UP|DOWN], GET_DOM, SCREENSHOT[PATH=…], WAIT[SECONDS=…], ASK USER HELP[TEXT=…]
(OpenTable 실험에서 사용한 액션 셋 그대로.  ￼)
	•	DOM 중간 표현/ID 태깅: 페이지의 인터랙티브 요소에 data-agentq-id 부여, 요약과 함께 상위 200개 요소를 구조화해 제공.
	•	Critic 기반 후보 랭킹 + MCTS‑lite 선택기(깊이 1): Thought 단계에서 3~6개 후보 커맨드를 만들고 critic 프롬프트로 점수화(JSON) → 이전 단계의 간단한 **Q 통계(q_stats)**와 결합해 최종 액션 1개를 선택. (검색/학습 없이 추론 시 탐색만 사용)
	•	OpenTable 휴리스틱 평가(옵션): URL이 opentable.com일 때 도메인 키워드(“Complete reservation”, “Reservation confirmed” 등)가 DOM 텍스트에 나타나면 성공으로 판정. (논문은 GPT‑4V 평가를 썼지만 여기서는 LLM 학습 없이 간단한 규칙 평가만 사용.  ￼)

⸻

2) 멀티‑파일 패치 지시문

아래 지시를 파일별로 적용하세요. (코드 블록 전체를 그대로 삽입/교체)

A. agentq/state.py

1) ActionType 확장
	•	class ActionType(Enum):에서 마지막 SCROLL = "scroll" 바로 아래 줄에 추가:
    ```python
    SUBMIT = "submit"
    CLEAR = "clear"
    ASK_USER_HELP = "ask_user_help"
    ```
2) AgentState(TypedDict) 필드 추가
	•	page_content: Optional[str] # 현재 페이지 내용 (요약) 바로 아래에 추가:
```python
    # 탐색/선택 보조 정보
    candidate_commands: Optional[List[str]]
    critic_scores: Optional[List[float]]
    q_stats: Optional[Dict[str, Any]]
    last_command: Optional[str]
    status: Optional[str]
```
3) create_initial_state(...) 반환 딕셔너리에서 page_content=None, 다음 줄에 추가:
```python
            candidate_commands=[],
            critic_scores=[],
            q_stats={},
            last_command=None,
            status=None,
```
B. agentq/prompt.py

1) SYSTEM_PROMPTS[“thought”] 교체
	•	기존 "thought": """You are AgentQ, ...""" 전체를 아래로 교체:
```python
    "thought": """You are AgentQ, an advanced AI agent that can interact with web pages.

Context:
- Objective: {objective}
- Current URL: {current_url}
- Page Title: {page_title}
- Plan: {plan}
- Loop: {loop_count}/{max_loops}
- Previous observation: {observation}

Action grammar (use exactly one line per command):
- GOTO [URL=<http(s)://...>]
- SEARCH [TEXT=<query>]
- CLICK [ID=<data-agentq-id>]
- TYPE [ID=<data-agentq-id>] [TEXT=<free text>]
- SUBMIT [ID=<data-agentq-id>]
- CLEAR [ID=<data-agentq-id>]
- SCROLL [UP|DOWN]
- GET_DOM
- SCREENSHOT [PATH=<filename.png>]
- WAIT [SECONDS=<int>]
- ASK USER HELP [TEXT=<question>]

Respond STRICTLY in the following format:

PLAN:
<one paragraph plan>

THOUGHT:
<concise reasoning for selecting NEXT action candidates>

COMMANDS:
- <candidate command #1>
- <candidate command #2>
- <candidate command #3>

STATUS:
CONTINUE""",
```
2) SYSTEM_PROMPTS에 "critic" 추가
	•	SYSTEM_PROMPTS 딕셔너리의 마지막 항목 뒤에 콤마 포함하여 추가:
```python
    "critic": """You are AgentQ's self-critic. Rank candidate COMMANDS for the next step.

You are given:
- Objective: {objective}
- Current URL: {current_url}
- Page Title: {page_title}
- Scratchpad: {scratchpad}
- Latest observation: {observation}

Scoring rules (0.0~1.0):
- + Progress toward goal, low-risk, minimal detours
- + Uses visible interactive element IDs when clicking/typing
- - Dead-ends (login, cookie banners) unless necessary
- - Irreversible or off-domain navigation without reason

Return ONLY a compact JSON array:
[
  {"cmd": "<verbatim command>", "score": 0.0~1.0, "rationale": "<short>"},
  ...
]
"""
```
3) FEW_SHOT_EXAMPLES[“thought”] 업데이트(간단 예시 1개로 교체)
	•	기존 FEW_SHOT_EXAMPLES["thought"] 리스트를 아래로 교체:
```python
    "thought": [
        {
            "context": "OpenTable home loaded; need to search restaurant",
            "reasoning": "Propose multiple safe next-steps using allowed grammar",
            "action": "Example output:\n\nPLAN:\nSearch for the restaurant then pick a time.\n\nTHOUGHT:\nWe should search first.\n\nCOMMANDS:\n- TYPE [ID=search_input] [TEXT=Cecconi's New York]\n- CLICK [ID=submit_search]\n- GET_DOM\n\nSTATUS:\nCONTINUE"
        }
    ],
```

⸻

C. agentq/prompt_utils.py

1) PromptBuilder에 critic 프롬프트 빌더 추가
	•	class PromptBuilder 안에 아래 메서드 추가:
```python
    def build_critic_prompt(self, state: AgentState) -> str:
        context = format_state_for_prompt(state)
        return build_prompt_with_examples(
            "critic",
            include_examples=False,
            **context
        )
```
2) 출력 블록 파서 및 커맨드 파서 추가
	•	파일 하단의 extract_action_from_response 위쪽에 다음 함수들을 추가:
```python
def split_output_blocks(response: str) -> Dict[str, str]:
    import re
    blocks = {"PLAN": "", "THOUGHT": "", "COMMANDS": "", "STATUS": ""}
    pattern = r'(?mi)^(PLAN|THOUGHT|COMMANDS|STATUS)\s*:\s*'
    parts = re.split(pattern, response)
    # parts = ["<head>", "PLAN", "<...>", "THOUGHT", "<...>", ...]
    if len(parts) < 3:
        return blocks
    current = None
    buffer = []
    for p in parts:
        if p in blocks:
            if current:
                blocks[current] = "\n".join(buffer).strip()
                buffer = []
            current = p
        else:
            buffer.append(p.strip())
    if current:
        blocks[current] = "\n".join(buffer).strip()
    return blocks

def extract_commands_and_status(response: str):
    blocks = split_output_blocks(response)
    raw_cmds = []
    for line in blocks.get("COMMANDS","").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("-"):
            line = line[1:].strip()
        raw_cmds.append(line)
    status = blocks.get("STATUS","").strip().upper().splitlines()[0] if blocks.get("STATUS") else ""
    return [c for c in raw_cmds if c], status

def parse_command_line(cmd: str) -> Optional[Dict[str, Any]]:
    import re
    c = cmd.strip()
    # GOTO / NAVIGATE
    m = re.match(r'^(?:GOTO|NAVIGATE)\s*\[\s*URL\s*=\s*([^\]]+)\s*\]\s*$', c, re.I)
    if m:
        return {"type": "NAVIGATE", "target": m.group(1).strip()}
    # SEARCH (maps to SEARCH text then NAVIGATE google?q=)
    m = re.match(r'^SEARCH\s*\[\s*TEXT\s*=\s*(.+?)\s*\]\s*$', c, re.I)
    if m:
        return {"type": "SEARCH", "content": m.group(1).strip()}
    # CLICK
    m = re.match(r'^CLICK\s*\[\s*ID\s*=\s*([^\]]+)\s*\]\s*$', c, re.I)
    if m:
        return {"type": "CLICK", "target": m.group(1).strip(), "by": "agentq-id"}
    # TYPE
    m = re.match(r'^TYPE\s*\[\s*ID\s*=\s*([^\]]+)\s*\]\s*\[\s*TEXT\s*=\s*(.+?)\s*\]\s*$', c, re.I)
    if m:
        return {"type": "TYPE", "target": m.group(1).strip(), "content": m.group(2).strip(), "by": "agentq-id"}
    # SUBMIT
    m = re.match(r'^SUBMIT\s*\[\s*ID\s*=\s*([^\]]+)\s*\]\s*$', c, re.I)
    if m:
        return {"type": "SUBMIT", "target": m.group(1).strip(), "by": "agentq-id"}
    # CLEAR
    m = re.match(r'^CLEAR\s*\[\s*ID\s*=\s*([^\]]+)\s*\]\s*$', c, re.I)
    if m:
        return {"type": "CLEAR", "target": m.group(1).strip(), "by": "agentq-id"}
    # SCROLL
    m = re.match(r'^SCROLL\s*\[\s*(UP|DOWN)\s*\]\s*$', c, re.I)
    if m:
        return {"type": "SCROLL", "target": m.group(1).lower()}
    # GET_DOM
    m = re.match(r'^GET_DOM\s*$', c, re.I)
    if m:
        return {"type": "GET_DOM"}
    # SCREENSHOT
    m = re.match(r'^SCREENSHOT(?:\s*\[\s*PATH\s*=\s*([^\]]+)\s*\])?\s*$', c, re.I)
    if m:
        return {"type": "SCREENSHOT", "target": m.group(1).strip() if m.group(1) else "screenshot.png"}
    # WAIT
    m = re.match(r'^WAIT\s*\[\s*SECONDS\s*=\s*(\d+)\s*\]\s*$', c, re.I)
    if m:
        return {"type": "WAIT", "content": m.group(1).strip()}
    # ASK USER HELP
    m = re.match(r'^ASK\s*USER\s*HELP\s*\[\s*TEXT\s*=\s*(.+?)\s*\]\s*$', c, re.I)
    if m:
        return {"type": "ASK_USER_HELP", "content": m.group(1).strip()}
    return None
```
3) extract_action_from_response의 최상단에 다음 3줄을 추가 (기존 로직은 그대로 유지; 실패 시 폴백):
```python
    cmds, _ = extract_commands_and_status(response)
    if cmds:
        parsed = parse_command_line(cmds[0])
        if parsed: return parsed
```

⸻

D. agentq/playwright_helper.py

1) 인터랙티브 요소 인덱싱 및 ID 조작 유틸 추가
	•	파일 하단(편의 함수들 정의 아래)에 아래 함수들 추가:
```python
async def index_interactive_elements():
    try:
        page = await get_current_page()
        if not page:
            return []
        elements = await page.evaluate("""
            () => {
                function visible(el){
                    const r = el.getBoundingClientRect();
                    const cs = getComputedStyle(el);
                    return r.width > 0 && r.height > 0 && cs.visibility !== 'hidden' && cs.display !== 'none';
                }
                const nodes = Array.from(document.querySelectorAll(
                  'button, a[href], input, select, textarea, [role="button"]'
                )).filter(visible);
                let i = 0;
                nodes.forEach(el => { if(!el.dataset.agentqId){ el.dataset.agentqId = 'el_' + (++i); }});
                return nodes.slice(0,200).map(el => ({
                    id: el.dataset.agentqId,
                    tag: el.tagName.toLowerCase(),
                    role: el.getAttribute('role') || (el.tagName.toLowerCase()==='a' ? 'link' :
                         (el.tagName.toLowerCase()==='button' ? 'button' :
                         (['input','select','textarea'].includes(el.tagName.toLowerCase())?'input':''))),
                    text: (el.innerText||'').trim().slice(0,80),
                    placeholder: el.getAttribute('placeholder') || '',
                    type: el.getAttribute('type') || '',
                    name: el.getAttribute('name') || '',
                    href: el.getAttribute('href') || ''
                }));
            }
        """)
        return elements or []
    except Exception as e:
        print(f"❌ index_interactive_elements 오류: {e}")
        return []

async def click_by_agentq_id(agentq_id: str) -> bool:
    try:
        page = await get_current_page()
        if not page: return False
        return await page.evaluate("""(id)=>{
            const el = document.querySelector(`[data-agentq-id="${id}"]`);
            if(!el) return false;
            el.click(); return true;
        }""", agentq_id)
    except Exception as e:
        print(f"❌ click_by_agentq_id 오류: {e}")
        return False

async def set_input_by_agentq_id(agentq_id: str, text: str) -> bool:
    try:
        page = await get_current_page()
        if not page: return False
        return await page.evaluate("""(id, val)=>{
            const el = document.querySelector(`[data-agentq-id="${id}"]`);
            if(!el) return false;
            if('value' in el) el.value = val;
            el.dispatchEvent(new Event('input', {bubbles:true}));
            return true;
        }""", agentq_id, text)
    except Exception as e:
        print(f"❌ set_input_by_agentq_id 오류: {e}")
        return False

async def clear_by_agentq_id(agentq_id: str) -> bool:
    return await set_input_by_agentq_id(agentq_id, "")

async def submit_by_agentq_id(agentq_id: str) -> bool:
    try:
        page = await get_current_page()
        if not page: return False
        return await page.evaluate("""(id)=>{
            const el = document.querySelector(`[data-agentq-id="${id}"]`);
            if(!el) return false;
            if(el.form){ el.form.requestSubmit ? el.form.requestSubmit() : el.form.submit(); return true; }
            el.click(); return true;
        }""", agentq_id)
    except Exception as e:
        print(f"❌ submit_by_agentq_id 오류: {e}")
        return False

async def get_dom_snapshot():
    try:
        page = await get_current_page()
        if not page: return None
        elements = await index_interactive_elements()
        title = await page.title()
        url = page.url
        text = await page.evaluate("() => document.body.innerText.slice(0, 3000)")
        return {"title": title, "url": url, "content": text, "elements": elements}
    except Exception as e:
        print(f"❌ get_dom_snapshot 오류: {e}")
        return None
```

⸻

E. agentq/tools.py

1) 상단 import 확장
	•	기존 import 블록에 아래 항목 추가:
```python
    # 아래 2줄 추가
    from agentq.playwright_helper import (
        index_interactive_elements, get_dom_snapshot,
        click_by_agentq_id, set_input_by_agentq_id, submit_by_agentq_id, clear_by_agentq_id
    )
```
2) WebTool.extract_page_content 구현 교체
	•	기존 extract_page_content 함수 전체를 아래로 교체:
```python
    @staticmethod
    async def extract_page_content() -> Dict[str, Any]:
        """페이지 내용 추출 (요약 + 인터랙티브 요소 목록 포함)"""
        try:
            snapshot = await get_dom_snapshot()
            if not snapshot:
                return {
                    "success": False,
                    "message": "페이지에 접근할 수 없습니다",
                    "data": None
                }
            return {
                "success": True,
                "message": "페이지 내용 추출 완료",
                "data": snapshot
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"페이지 내용 추출 중 오류: {str(e)}",
                "data": None
            }
```
3) ToolExecutor.execute_action에 확장 액션 처리 추가
	•	함수 내부의 분기들 아래에 다음 분기들을 추가 (기존 CLICK/TYPE도 보강):
```python
            elif action_type == "CLICK":
                by = action.get("by", "")
                target = action.get("target", "")
                if by == "agentq-id":
                    ok = await click_by_agentq_id(target)
                    return {"success": ok, "message": f"클릭 {'성공' if ok else '실패'}(ID): {target}", "data": None}
                else:
                    return await self.web_tool.click(target)

            elif action_type == "TYPE":
                by = action.get("by", "")
                target = action.get("target", "")
                text = action.get("content", "")
                if by == "agentq-id":
                    ok = await set_input_by_agentq_id(target, text)
                    return {"success": ok, "message": f"입력 {'성공' if ok else '실패'}(ID): {text}", "data": None}
                else:
                    return await self.web_tool.type_text_input(target, text)

            elif action_type == "SUBMIT":
                target = action.get("target", "")
                ok = await submit_by_agentq_id(target)
                return {"success": ok, "message": f"제출 {'성공' if ok else '실패'}(ID): {target}", "data": None}

            elif action_type == "CLEAR":
                target = action.get("target", "")
                ok = await clear_by_agentq_id(target)
                return {"success": ok, "message": f"지우기 {'성공' if ok else '실패'}(ID): {target}", "data": None}

            elif action_type == "ASK_USER_HELP":
                # 도메인 상 위험/애매 상황 → 상위 레이어가 사용자 상호작용으로 처리
                text = action.get("content", "")
                return {"success": False, "message": f"사용자 확인 필요: {text}", "data": {"ask_user": True, "text": text}}
```

⸻

F. agentq/nodes.py

1) import 확장
	•	상단 import 중 from agentq.prompt_utils import ... 라인을 아래로 교체:
```python
from agentq.prompt_utils import (
    get_prompt_builder, ScratchpadManager,
    extract_action_from_response, extract_critique_decision, clean_response,
    extract_commands_and_status, parse_command_line
)
```
2) thought_node를 후보 생성+critic 랭킹+선택 로직으로 교체
	•	기존 async def thought_node(state: AgentState) -> Dict[str, Any]: 함수 본문 전체를 아래로 교체:
```python
async def thought_node(state: AgentState) -> Dict[str, Any]:
    """Thought 노드: 후보 생성 → critic 점수화 → 최종 액션 선택"""
    print(f"🤔 Thought 노드 실행 중... (루프 {state['loop_count'] + 1})")
    try:
        # 루프 카운트 증가
        state = increment_loop_count(state)

        # 최신 DOM 스냅샷 확보 (없을 때만)
        tool_executor = get_tool_executor()
        if not state.get("page_content"):
            snap = await tool_executor.extract_page_content()
            if snap.get("success") and isinstance(snap.get("data"), dict):
                d = snap["data"]
                state["current_url"] = d.get("url")
                state["page_title"] = d.get("title")
                state["page_content"] = (d.get("content") or "")[:500]
                state["observation"] = "페이지 내용 추출 완료"

        # 후보 커맨드 생성
        prompt_builder = get_prompt_builder()
        system_prompt = prompt_builder.build_thought_prompt(state)
        llm_manager = get_llm_manager()
        resp = await llm_manager.invoke_with_system(
            system_prompt=system_prompt,
            user_message="Propose multiple candidate COMMANDS for the very next step."
        )
        raw = clean_response(resp)
        cmds, status = extract_commands_and_status(raw)
        thought_text = split_output_blocks(raw).get("THOUGHT","") if 'split_output_blocks' in globals() else raw

        # critic으로 랭킹
        critic_prompt = prompt_builder.build_critic_prompt(state)
        critic_in = "Rank these commands:\n" + "\n".join([f"{i+1}. {c}" for i,c in enumerate(cmds)])
        critic_out = await llm_manager.invoke_with_system(critic_prompt, critic_in)
        import json
        scores = []
        try:
            parsed = json.loads(critic_out)
            if isinstance(parsed, list):
                for p in parsed:
                    if isinstance(p, dict) and "cmd" in p and "score" in p:
                        scores.append((p["cmd"], float(p["score"])))
        except Exception:
            pass
        # critic 출력이 불완전하면 균등 분배
        if not scores:
            scores = [(c, 0.5) for c in cmds]

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
        best_cmd = scored[0][1] if scored else (cmds[0] if cmds else "GET_DOM")
        action = parse_command_line(best_cmd) or extract_action_from_response(best_cmd)

        # 상태 업데이트
        state["thought"] = thought_text or "다음 행동 후보를 생성했습니다."
        state["candidate_commands"] = cmds
        state["critic_scores"] = [s for _, s in scores]
        state["status"] = status or "CONTINUE"
        state["last_command"] = best_cmd
        state["action"] = action

        # 스크래치패드 기록
        state = ScratchpadManager.add_thought(state, state["thought"])
        if action:
            state = ScratchpadManager.add_action(state, action)

        print(f"   후보 개수: {len(cmds)} / 선택: {best_cmd}")
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
```
3) action_node 성공 시 DOM 요약 반영 보강 (선택사항)
— 기존 코드가 GET_DOM에서 데이터를 반영하고 있어 필수 변경은 아닙니다. 그대로 두셔도 됩니다.

4) critique_node에 휴리스틱 완료 판정 + q_stats 업데이트 추가
	•	done = extract_critique_decision(response) 아래쪽에 다음 블록 추가:
```python
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
        try:
            cmd = state.get("last_command")
            if cmd:
                stats = state.get("q_stats") or {}
                ent = stats.get(cmd, {"Q": 0.0, "N": 0})
                reward = 1.0 if done else 0.0
                ent["N"] += 1
                ent["Q"] = ent["Q"] + (reward - ent["Q"]) / ent["N"]
                stats[cmd] = ent
                state["q_stats"] = stats
        except Exception:
            pass
```

⸻

G. (선택) agentq/restaurant_agent.py

필수는 아닙니다. OpenTable 안내/검색 함수는 그대로; 별도 휴리스틱 클래스가 필요하면 아래와 같이 추가할 수 있습니다.
```python
class OpenTableEvaluator:
    KEYWORDS_SUCCESS = ["reservation confirmed", "complete reservation", "you're all set"]
    @staticmethod
    def evaluate_text(text: str) -> bool:
        if not text: return False
        t = text.lower()
        return any(k in t for k in OpenTableEvaluator.KEYWORDS_SUCCESS)
```

⸻

3) 동작 방식
	1.	Thought: LLM이 논문 규격대로 PLAN/THOUGHT/COMMANDS/STATUS를 생성 →
COMMANDS 블록에서 3~6개 후보 명령을 파싱.
	2.	Critic: 별도 critic 프롬프트로 각 후보를 0~1 점수로 JSON 랭킹.
	3.	선택: critic 점수와 세션 내부 q_stats(최근 보상 추정치)를 결합해 1개 명령을 선택(MCTS‑lite, 깊이 1).
	4.	Action: 선택된 명령을 파싱해 ToolExecutor가 실행.
CLICK/TYPE/SUBMIT/CLEAR는 data-agentq-id 기반으로 동작.
	5.	Critique: LLM 판단 + (OpenTable일 때) 휴리스틱 키워드로 성공 판단. 세션 내 Q 통계 업데이트.

논문의 입력/출력 포맷과 액션 셋을 그대로 따릅니다. (입·출력 구조: Fig.2, 액션 정의: 실험 섹션)  ￼  ￼

⸻

4) 빠른 점검 방법
```python
from agentq.graph import AgentQExecutor
import asyncio

async def main():
    ex = AgentQExecutor()
    ex.compile()
    # 예) OpenTable 테스트 목적 입력 (실제 실행 전 Chrome/Playwright 연결 필수)
    final_state = await ex.execute("OpenTable에서 'Cote Korean Steakhouse' 2명 내일 저녁 7시 예약 가능 시간 확인")
    print(final_state.get("explanation"))

asyncio.run(main())
```
	•	Playwright 연결: 현재 헬퍼는 이미 연결 유틸이 포함되어 있습니다. 크롬 디버깅 포트 또는 headless 브라우저를 사용하세요.
	•	확인 포인트: 로그에 COMMANDS 후보 수, critic 선택 결과, data-agentq-id가 보이는지.

⸻

5) 한계 및 메모
	•	DPO/강화학습은 제외되어 있습니다. (정책 학습 없이 추론 시 탐색/자기평가만 사용)
	•	MCTS‑lite(깊이 1) 구조라 롤백/멀티 컨텍스트 시뮬은 하지 않습니다. 그래도 critic 점수와 간단한 Q 통계로 반복 시 수렴 경향을 갖습니다(세션 한정).
	•	사이트마다 data-agentq-id 할당 정확도는 DOM 구조/표시여부에 영향을 받습니다. 필요하면 index_interactive_elements()의 CSS 후보/가시성 조건을 조정하세요.

⸻

6) 왜 이렇게 했나 (논문 대비)
	•	출력/행동 표준화는 Figure 2의 포맷을 그대로 구현했습니다.  ￼
	•	행동 셋은 OpenTable 실험에서 사용된 것과 동일합니다.  ￼
	•	AI critic은 논문의 “self‑critique/AI feedback” 아이디어를 추론 시 점수화에 한정해 반영했습니다(학습 제외).  ￼
	•	OpenTable 평가는 논문처럼 GPT‑4V 평가는 쓰지 않고(학습·비용 이슈), 간단한 규칙 기반 체크만 추가했습니다.  ￼

⸻

필요하시면 이 패치를 자동으로 적용하도록 도와드릴 수 있습니다. 우선 위 변경을 반영해 보시고, 실행 중 에러/엣지 케이스가 보이면 로그와 함께 알려주세요.
