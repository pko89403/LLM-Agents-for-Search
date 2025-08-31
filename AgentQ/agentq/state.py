"""
LangGraph AgentState 정의
"""

from typing import List, Optional, Dict, Any, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from enum import Enum


class ActionType(Enum):
    """액션 타입 정의"""
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SCREENSHOT = "screenshot"
    GET_DOM = "get_dom"
    SEARCH = "search"
    WAIT = "wait"
    SCROLL = "scroll"
    SUBMIT = "submit"
    CLEAR = "clear"
    ASK_USER_HELP = "ask_user_help"


class Action(BaseModel):
    """실행할 액션 정의"""
    type: ActionType
    target: Optional[str] = None  # CSS 선택자, URL 등
    content: Optional[str] = None  # 입력할 텍스트, 검색어 등
    timeout: Optional[int] = 5000  # 타임아웃 (밀리초)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "target": self.target,
            "content": self.content,
            "timeout": self.timeout
        }


class AgentState(TypedDict):
    """LangGraph AgentState 정의"""
    # 기본 정보
    user_input: str  # 사용자 질문/명령
    objective: str   # 목표 (user_input과 동일하거나 정제된 버전)

    # AgentQ 루프 상태
    plan: Optional[str]  # 계획 (Plan 단계 결과)
    thought: Optional[str]  # 추론 내용 (Thought 단계 결과)
    action: Optional[Dict[str, Any]]  # 실행할 액션 (Action 객체의 dict 형태)
    observation: Optional[str]  # 환경으로부터 얻은 관찰 결과
    explanation: Optional[str]  # 설명/내부 해설 (Explanation 단계 결과)

    # 제어 플래그
    done: bool  # 완료 여부 (Critique 판단)
    loop_count: int  # 루프 반복 횟수
    max_loops: int  # 최대 루프 횟수

    # 웹 상태
    current_url: Optional[str]  # 현재 페이지 URL
    page_title: Optional[str]   # 현재 페이지 제목
    page_content: Optional[str] # 현재 페이지 내용 (요약)

    # 탐색/선택 보조 정보
    candidate_commands: Optional[List[str]]
    critic_scores: Optional[List[float]]
    q_stats: Optional[Dict[str, Any]]
    last_command: Optional[str]
    status: Optional[str]
    min_loops: int
    no_progress_streak: int
    last_progress_fingerprint: Optional[str]

    # 메시지 히스토리 (LangGraph 메시지 관리)
    messages: Annotated[List[Dict[str, Any]], add_messages]

    # 스크래치패드 (중간 결과 저장)
    scratchpad: List[str]

    # 에러 정보
    last_error: Optional[str]
    error_count: int

    # 메타데이터
    session_id: Optional[str]
    start_time: Optional[str]

    @classmethod
    def create_initial_state(
        cls,
        user_input: str,
        max_loops: int = 5,
        session_id: Optional[str] = None
    ) -> "AgentState":
        """초기 상태 생성"""
        from datetime import datetime

        return cls(
            user_input=user_input,
            objective=user_input,
            plan=None,
            thought=None,
            action=None,
            observation=None,
            explanation=None,
            done=False,
            loop_count=0,
            max_loops=max_loops,
            current_url=None,
            page_title=None,
            page_content=None,
            candidate_commands=[],
            critic_scores=[],
            q_stats={},
            last_command=None,
            status=None,
            min_loops=3,
            no_progress_streak=0,
            last_progress_fingerprint=None,
            messages=[],
            scratchpad=[],
            last_error=None,
            error_count=0,
            session_id=session_id,
            start_time=datetime.now().isoformat()
        )


def add_to_scratchpad(state: AgentState, content: str) -> AgentState:
    """스크래치패드에 내용 추가"""
    state["scratchpad"].append(content)
    return state


def get_scratchpad_content(state: AgentState, max_entries: int = 10) -> str:
    """스크래치패드 내용을 문자열로 반환"""
    recent_entries = state["scratchpad"][-max_entries:]
    return "\n".join(f"- {entry}" for entry in recent_entries)


def increment_loop_count(state: AgentState) -> AgentState:
    """루프 카운트 증가"""
    state["loop_count"] += 1
    return state


def add_error(state: AgentState, error: str) -> AgentState:
    """에러 정보 추가"""
    state["last_error"] = error
    state["error_count"] += 1
    return state

def clear_error(state: AgentState) -> AgentState:
    """에러 정보 초기화"""
    state["last_error"] = None
    return state
