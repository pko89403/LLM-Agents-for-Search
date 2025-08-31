"""
LangGraph 그래프 정의 및 에이전트 실행 로직
"""

from langgraph.graph import StateGraph, START, END
from agentq.state import AgentState
from agentq.nodes import (
    plan_node, thought_node, action_node, 
    explanation_node, critique_node, should_continue
)


def create_agentq_graph() -> StateGraph:
    """AgentQ LangGraph 생성"""
    
    # StateGraph 초기화
    graph = StateGraph(AgentState)
    
    # 노드 추가
    graph.add_node("plan_node", plan_node)
    graph.add_node("thought_node", thought_node)
    graph.add_node("action_node", action_node)
    graph.add_node("explanation_node", explanation_node)
    graph.add_node("critique_node", critique_node)
    
    # 엣지 연결
    # START → Plan (처음 한 번만)
    graph.add_edge(START, "plan_node")
    
    # Plan → Thought (계획 수립 후 첫 번째 사고)
    graph.add_edge("plan_node", "thought_node")
    
    # Thought → Action (사고 후 행동)
    graph.add_edge("thought_node", "action_node")
    
    # Action → Explanation (행동 후 설명)
    graph.add_edge("action_node", "explanation_node")
    
    # Explanation → Critique (설명 후 평가)
    graph.add_edge("explanation_node", "critique_node")
    
    # Critique → 조건부 분기 (완료 여부에 따라)
    graph.add_conditional_edges(
        "critique_node",
        should_continue,
        {
            "thought": "thought_node",  # 계속하는 경우 다시 사고 단계로
            "end": END                 # 완료된 경우 종료
        }
    )
    
    return graph


class AgentQExecutor:
    """AgentQ 실행기"""
    
    def __init__(self):
        self.graph = create_agentq_graph()
        self.compiled_graph = None
    
    def compile(self):
        """그래프 컴파일"""
        self.compiled_graph = self.graph.compile()
        print("✅ AgentQ 그래프가 컴파일되었습니다.")
    
    async def execute(
        self, 
        user_input: str, 
        max_loops: int = 5,
        session_id: str = None
    ) -> AgentState:
        """AgentQ 실행"""
        
        if not self.compiled_graph:
            self.compile()
        
        # 초기 상태 생성
        initial_state = AgentState.create_initial_state(
            user_input=user_input,
            max_loops=max_loops,
            session_id=session_id
        )
        
        print(f"🎯 AgentQ 실행 시작: {user_input}")
        print("=" * 60)
        
        try:
            # 그래프 실행
            final_state = await self.compiled_graph.ainvoke(initial_state)
            
            print("\n" + "=" * 60)
            print("🎉 AgentQ 실행 완료!")
            print(f"📊 총 루프 횟수: {final_state['loop_count']}")
            print(f"✅ 완료 상태: {'성공' if final_state['done'] else '미완료'}")
            
            if final_state.get("explanation"):
                print(f"📝 최종 설명: {final_state['explanation']}")
            
            return final_state
            
        except Exception as e:
            print(f"\n❌ AgentQ 실행 중 오류 발생: {str(e)}")
            # 오류 상태로 반환
            initial_state["done"] = True
            initial_state["last_error"] = str(e)
            initial_state["explanation"] = f"실행 중 오류가 발생했습니다: {str(e)}"
            return initial_state
    
    async def stream_execute(
        self, 
        user_input: str, 
        max_loops: int = 5,
        session_id: str = None
    ):
        """AgentQ 스트리밍 실행 (중간 과정 실시간 출력)"""
        
        if not self.compiled_graph:
            self.compile()
        
        # 초기 상태 생성
        initial_state = AgentState.create_initial_state(
            user_input=user_input,
            max_loops=max_loops,
            session_id=session_id
        )
        
        print(f"🎯 AgentQ 스트리밍 실행 시작: {user_input}")
        print("=" * 60)
        
        try:
            # 스트리밍 실행
            async for event in self.compiled_graph.astream(initial_state):
                # 각 노드 실행 결과를 실시간으로 출력
                for node_name, node_output in event.items():
                    print(f"\n🔄 노드 '{node_name}' 완료")
                    if isinstance(node_output, dict):
                        for key, value in node_output.items():
                            if value and len(str(value)) > 100:
                                print(f"   {key}: {str(value)[:100]}...")
                            else:
                                print(f"   {key}: {value}")
                    print("-" * 40)
            
            print("\n" + "=" * 60)
            print("🎉 AgentQ 스트리밍 실행 완료!")
            
        except Exception as e:
            print(f"\n❌ AgentQ 스트리밍 실행 중 오류 발생: {str(e)}")
    
    def get_graph_visualization(self) -> str:
        """그래프 구조 시각화 (텍스트)"""
        return """
AgentQ Graph Structure:
======================

START
  ↓
PLAN (계획 수립)
  ↓
THOUGHT (사고 과정)
  ↓
ACTION (액션 실행)
  ↓
EXPLANATION (결과 해석)
  ↓
CRITIQUE (완료 여부 평가)
  ↓
  ├─ CONTINUE → THOUGHT (루프 반복)
  └─ COMPLETE → END (종료)

노드 설명:
- PLAN: 전체 작업 계획 수립 (한 번만 실행)
- THOUGHT: 현재 상황에서 다음 행동 결정
- ACTION: 웹 상호작용 실행 (클릭, 입력, 검색 등)
- EXPLANATION: 액션 결과 해석 및 의미 분석
- CRITIQUE: 목표 달성 여부 판단 및 루프 제어
"""


# 전역 실행기 인스턴스
_executor: AgentQExecutor = None

def get_agentq_executor() -> AgentQExecutor:
    """AgentQ 실행기 싱글톤 인스턴스 반환"""
    global _executor
    if _executor is None:
        _executor = AgentQExecutor()
    return _executor