from typing import List, Literal, Optional, Any
from typing_extensions import TypedDict, NotRequired


class LaserState(TypedDict):
   """LASER 에이전트의 전체 상태를 나타내는 TypedDict입니다."""

   # 입력/맥락
   user_instruction: str

   # 환경 상호작용
   obs: str                    # 현재 관찰 (html/text)
   url: Optional[str]
   current_laser_state: Literal["Search", "Result", "Item", "Stopping"]

   # 이력/메모리
   step_count: int
   thought_history: NotRequired[List[str]]
   action_history: NotRequired[List[str]]
   memory_buffer: NotRequired[List[dict]]   # 후보 아이템이나 유망 링크

   # 노드가 결정한 전이 및 산출물
   route: NotRequired[Literal["to_result", "to_item", "to_search", "stay_result", "stay_item", "to_stop"]]
   last_action: NotRequired[str]            # env에 보낸 raw action 문자열
   selected_item: NotRequired[dict]         # 최종 아이템(Stopping 용)
   info: NotRequired[dict]

   # 내부 핸들러 (환경 인스턴스 등)
   # laser_agent_dev_guide.md의 스켈레톤 코드에서 `_env`를 상태에 포함하여 노드에 전달합니다.
   _env: NotRequired[Any]
