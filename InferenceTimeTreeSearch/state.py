'''
LangGraph의 AgentState를 WebShop ITTS 에이전트에 맞게 재정의합니다.
guide.webshop.md의 2, 8.2 섹션을 참조합니다.
'''
from typing import Any, List, Optional, Dict, TypedDict, Tuple
from dataclasses import dataclass, field

# --- WebShop 환경 데이터 클래스 (guide.webshop.md 8.2) ---

@dataclass
class Product:
    """WebShop 상품 데이터 모델"""
    id: str
    title: str
    price: float
    rating: Optional[float] = None
    brand: Optional[str] = None
    attrs: Optional[Dict[str, Any]] = None

@dataclass
class Observation:
    """WebShop 환경 관측 데이터 모델"""
    url: Optional[str]
    query: Optional[str]
    page: int
    sort: Optional[str]
    filters: Dict[str, str]
    results: List[Product]
    cart: List[Product]
    
    # 사용자의 설명에 따라 추가된 필드
    available_actions: Dict[str, Dict[str, str]]  # {'has_search_bar': bool, 'clickables': Dict[str, str] (text -> url/action)}
    
    html: Optional[str] = None

# --- ITTS 에이전트 상태 (guide.webshop.md 2) ---

@dataclass(order=True)
class PrioritizedItem:
    """우선순위 큐에 저장될 아이템"""
    priority: float
    state: Any = field(compare=False)

class SearchState(TypedDict):
    """
    ITTS(Inference-time Tree Search) 에이전트의 상태
    """
    # 목표 및 검색 예산
    goal: str
    max_steps: int
    search_counter: int
    branching: int
    budget: int

    # 현재 상태 정보
    observation: Observation
    action_history: List[str]
    
    # ITTS 관련
    frontier: List[PrioritizedItem]  # 실제 구현은 heapq 래퍼 클래스 사용
    
    # 최고 상태 추적
    best_state: Optional[Any] # 가장 점수가 높았던 상태 (SearchState의 스냅샷)
    best_score: float
    
    # 종료 조건
    done: bool
    final_answer: Optional[str]
