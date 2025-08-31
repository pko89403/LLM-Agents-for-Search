"""
AgentQ 데이터 모델 정의
"""

from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel


class ActionType(Enum):
    """액션 타입 정의"""
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SCREENSHOT = "screenshot"
    GET_DOM = "get_dom"
    SEARCH = "search"
    WAIT = "wait"


class Action(BaseModel):
    """실행할 액션 정의"""
    type: ActionType
    target: Optional[str] = None  # CSS 선택자, URL 등
    content: Optional[str] = None  # 입력할 텍스트, 검색어 등
    timeout: Optional[int] = 5000  # 타임아웃 (밀리초)


class AgentState(BaseModel):
    """AgentQ 상태 모델"""
    user_input: str  # 사용자 질문/명령
    plan: Optional[str] = None  # 계획 (Plan 단계 결과)
    thought: Optional[str] = None  # 추론 내용 (Thought 단계 결과)
    action: Optional[Action] = None  # 실행할 액션
    observation: Optional[str] = None  # 환경으로부터 얻은 관찰 결과
    explanation: Optional[str] = None  # 설명/내부 해설
    done: bool = False  # 완료 여부 (Critique 판단)
    loop_count: int = 0  # 루프 반복 횟수
    max_loops: int = 5  # 최대 루프 횟수


class AgentResponse(BaseModel):
    """에이전트 응답 모델"""
    success: bool
    message: str
    state: Optional[AgentState] = None
    error: Optional[str] = None