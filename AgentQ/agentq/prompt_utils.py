"""
LLM 프롬프트 구성 및 스크래치패드 관리 유틸리티
"""

import re
from typing import Dict, List, Optional, Any
from agentq.state import AgentState, get_scratchpad_content, add_to_scratchpad
from agentq.prompt import build_prompt_with_examples, format_state_for_prompt


class PromptBuilder:
    """프롬프트 구성 도우미 클래스"""
    
    def __init__(self, include_examples: bool = True):
        self.include_examples = include_examples
    
    def build_plan_prompt(self, state: AgentState) -> str:
        """Plan 단계 프롬프트 구성"""
        context = format_state_for_prompt(state)
        return build_prompt_with_examples(
            "plan", 
            include_examples=self.include_examples,
            **context
        )
    
    def build_thought_prompt(self, state: AgentState) -> str:
        """Thought 단계 프롬프트 구성"""
        context = format_state_for_prompt(state)
        return build_prompt_with_examples(
            "thought",
            include_examples=self.include_examples,
            **context
        )
    
    def build_explanation_prompt(self, state: AgentState) -> str:
        """Explanation 단계 프롬프트 구성"""
        context = format_state_for_prompt(state)
        return build_prompt_with_examples(
            "explanation",
            include_examples=self.include_examples,
            **context
        )
    
    def build_critique_prompt(self, state: AgentState) -> str:
        """Critique 단계 프롬프트 구성"""
        context = format_state_for_prompt(state)
        return build_prompt_with_examples(
            "critique",
            include_examples=self.include_examples,
            **context
        )

    def build_critic_prompt(self, state: AgentState) -> str:
        context = format_state_for_prompt(state)
        return build_prompt_with_examples(
            "critic",
            include_examples=False,
            **context
        )


class ScratchpadManager:
    """스크래치패드 관리 클래스"""
    
    @staticmethod
    def add_plan(state: AgentState, plan: str) -> AgentState:
        """계획을 스크래치패드에 추가"""
        entry = f"[PLAN] {plan}"
        return add_to_scratchpad(state, entry)
    
    @staticmethod
    def add_thought(state: AgentState, thought: str) -> AgentState:
        """사고 과정을 스크래치패드에 추가"""
        entry = f"[THOUGHT-{state['loop_count']}] {thought}"
        return add_to_scratchpad(state, entry)
    
    @staticmethod
    def add_action(state: AgentState, action: Dict[str, Any]) -> AgentState:
        """액션을 스크래치패드에 추가"""
        action_str = f"{action.get('type', 'UNKNOWN')}"
        if action.get('target'):
            action_str += f" -> {action['target']}"
        if action.get('content'):
            action_str += f" ({action['content']})"
        
        entry = f"[ACTION-{state['loop_count']}] {action_str}"
        return add_to_scratchpad(state, entry)
    
    @staticmethod
    def add_observation(state: AgentState, observation: str) -> AgentState:
        """관찰 결과를 스크래치패드에 추가"""
        entry = f"[OBSERVATION-{state['loop_count']}] {observation[:100]}..."
        return add_to_scratchpad(state, entry)
    
    @staticmethod
    def add_explanation(state: AgentState, explanation: str) -> AgentState:
        """설명을 스크래치패드에 추가"""
        entry = f"[EXPLANATION-{state['loop_count']}] {explanation}"
        return add_to_scratchpad(state, entry)
    
    @staticmethod
    def add_critique(state: AgentState, critique: str, done: bool) -> AgentState:
        """평가 결과를 스크래치패드에 추가"""
        status = "COMPLETE" if done else "CONTINUE"
        entry = f"[CRITIQUE-{state['loop_count']}] {status} - {critique}"
        return add_to_scratchpad(state, entry)
    
    @staticmethod
    def get_formatted_scratchpad(state: AgentState, max_entries: int = 10) -> str:
        """포맷된 스크래치패드 내용 반환"""
        return get_scratchpad_content(state, max_entries)

def split_output_blocks(response: str) -> Dict[str, str]:
    import re
    blocks = {"PLAN": "", "THOUGHT": "", "COMMANDS": "", "STATUS": ""}
    pattern = r'(?mi)^(PLAN|THOUGHT|COMMANDS|STATUS)\s*:\s*'
    parts = re.split(pattern, response)
    # parts = ["<head>", "PLAN", "<...", "THOUGHT", "<...", ...]
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
    for line in blocks.get("COMMANDS","" ).splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("-"):
            line = line[1:].strip()
        raw_cmds.append(line)
    status = blocks.get("STATUS","" ).strip().upper().splitlines()[0] if blocks.get("STATUS") else ""
    return [c for c in raw_cmds if c], status

def parse_command_line(cmd: str) -> Optional[Dict[str, Any]]:
    import re
    c = cmd.strip()
    # GOTO / NAVIGATE
    m = re.match(r'^(?:GOTO|NAVIGATE)\s*\[\s*URL\s*=\s*([^\\]+)\s*\]\s*$', c, re.I)
    if m:
        return {"type": "NAVIGATE", "target": m.group(1).strip()}
    # SEARCH (maps to SEARCH text then NAVIGATE google?q=)
    m = re.match(r'^SEARCH\s*\[\s*TEXT\s*=\s*(.+?)\s*\]\s*$', c, re.I)
    if m:
        return {"type": "SEARCH", "content": m.group(1).strip()}
    # CLICK
    m = re.match(r'^CLICK\s*\[\s*ID\s*=\s*([^\\]+)\s*\]\s*$', c, re.I)
    if m:
        return {"type": "CLICK", "target": m.group(1).strip(), "by": "agentq-id"}
    # TYPE
    m = re.match(r'^TYPE\s*\[\s*ID\s*=\s*([^\\]+)\s*\]\s*\[\s*TEXT\s*=\s*(.+?)\s*\]\s*$', c, re.I)
    if m:
        return {"type": "TYPE", "target": m.group(1).strip(), "content": m.group(2).strip(), "by": "agentq-id"}
    # SUBMIT
    m = re.match(r'^SUBMIT\s*\[\s*ID\s*=\s*([^\\]+)\s*\]\s*$', c, re.I)
    if m:
        return {"type": "SUBMIT", "target": m.group(1).strip(), "by": "agentq-id"}
    # CLEAR
    m = re.match(r'^CLEAR\s*\[\s*ID\s*=\s*([^\\]+)\s*\]\s*$', c, re.I)
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
    m = re.match(r'^SCREENSHOT(?:[\[\s*PATH\s*=\s*([^\\]+)\s*\])?\s*$', c, re.I)
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

def extract_action_from_response(response: str) -> Optional[Dict[str, Any]]:
    """LLM 응답에서 액션 추출"""
    cmds, _ = extract_commands_and_status(response)
    if cmds:
        parsed = parse_command_line(cmds[0])
        if parsed: return parsed
    import re
    
    # 액션 패턴 매칭
    patterns = {
        'NAVIGATE': r'NAVIGATE[:\s]+(.+)',
        'SEARCH': r'SEARCH[:\s]+(.+)',
        'CLICK': r'CLICK[:\s]+(.+)',
        'TYPE': r'TYPE[:\s]+(.+?)\s*\\|\\|\s*(.+)',
        'GET_DOM': r'GET_DOM',
        'WAIT': r'WAIT[:\s]+(\d+)',
        'SCROLL': r'SCROLL[:\s]+(up|down)'
    }
    
    for action_type, pattern in patterns.items():
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            action = {"type": action_type}
            
            if action_type == 'TYPE':
                action["target"] = match.group(1).strip()
                action["content"] = match.group(2).strip()
            elif action_type in ['NAVIGATE', 'CLICK', 'SEARCH']:
                action["target"] = match.group(1).strip()
            elif action_type == 'WAIT':
                action["content"] = match.group(1).strip()
            elif action_type == 'SCROLL':
                action["target"] = match.group(1).strip()
            
            return action
    
    return None

def extract_critique_decision(response: str) -> bool:
    """Critique 응답에서 완료 여부 추출"""
    response_upper = response.upper()
    
    if "COMPLETE" in response_upper:
        return True
    elif "CONTINUE" in response_upper:
        return False
    
    # 키워드 기반 추론
    complete_keywords = ["완료", "끝", "성공", "달성", "충분"]
    continue_keywords = ["계속", "더", "추가", "필요", "부족"]
    
    complete_score = sum(1 for keyword in complete_keywords if keyword in response)
    continue_score = sum(1 for keyword in continue_keywords if keyword in response)
    
    return complete_score > continue_score


def clean_response(response: str) -> str:
    """LLM 응답 정리"""
    # 불필요한 마크다운 제거
    response = re.sub(r'```[a-zA-Z]*\n?', '', response)
    response = re.sub(r'\*\*(.+?)\*\*', r'\1', response)
    response = re.sub(r'\*(.+?)\*', r'\1', response)
    
    # 여러 줄바꿈을 하나로
    response = re.sub(r'\n\s*\n', '\n', response)
    
    return response.strip()


# 전역 프롬프트 빌더 인스턴스
_prompt_builder: Optional[PromptBuilder] = None

def get_prompt_builder() -> PromptBuilder:
    """프롬프트 빌더 싱글톤 인스턴스 반환"""
    global _prompt_builder
    if _prompt_builder is None:
        _prompt_builder = PromptBuilder()
    return _prompt_builder