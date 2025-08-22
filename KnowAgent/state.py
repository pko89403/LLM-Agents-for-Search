from typing import List, Literal, Optional, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    scratchpad: str
    step: int
    action_type: Optional[Literal["Retrieve", "Search", "Lookup", "Finish"]]
    argument: Optional[str]
    last_passages: List[str]
    finished: bool
    answer: Optional[str]
