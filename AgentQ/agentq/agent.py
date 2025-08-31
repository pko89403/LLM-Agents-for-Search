"""
AgentQ 기본 에이전트 구현
Plan → Thought → Action → Explanation → Critique 루프
"""

import asyncio
from typing import Optional, Dict, Any
from agentq.models import AgentState, Action, ActionType, AgentResponse
from agentq.skills import execute_action, get_page_info, extract_text_content


class AgentQ:
    """AgentQ 기본 에이전트"""
    
    def __init__(self):
        self.state: Optional[AgentState] = None
    
    async def execute(self, user_input: str) -> AgentResponse:
        """
        사용자 입력을 받아 AgentQ 루프 실행
        
        Args:
            user_input: 사용자 질문/명령
        
        Returns:
            AgentResponse: 실행 결과
        """
        try:
            # 상태 초기화
            self.state = AgentState(user_input=user_input)
            
            print(f"🎯 목표: {user_input}")
            print("=" * 50)
            
            # Plan 단계 (한 번만 실행)
            await self._plan_step()
            
            # 메인 루프: Thought → Action → Explanation → Critique
            while not self.state.done and self.state.loop_count < self.state.max_loops:
                self.state.loop_count += 1
                print(f"\n🔄 루프 {self.state.loop_count}")
                print("-" * 30)
                
                # Thought 단계
                await self._thought_step()
                
                # Action 단계
                await self._action_step()
                
                # Explanation 단계
                await self._explanation_step()
                
                # Critique 단계
                await self._critique_step()
                
                if self.state.done:
                    break
                    
                # 다음 루프를 위한 짧은 대기
                await asyncio.sleep(1)
            
            # 최종 결과
            if self.state.done:
                return AgentResponse(
                    success=True,
                    message=self.state.explanation or "작업이 완료되었습니다.",
                    state=self.state
                )
            else:
                return AgentResponse(
                    success=False,
                    message=f"최대 루프 횟수({self.state.max_loops})에 도달했습니다.",
                    state=self.state
                )
                
        except Exception as e:
            return AgentResponse(
                success=False,
                message="실행 중 오류가 발생했습니다.",
                error=str(e),
                state=self.state
            )
    
    async def _plan_step(self):
        """Plan 단계: 전체 계획 수립"""
        print("📋 Plan 단계: 계획 수립 중...")
        
        # 간단한 계획 수립 로직 (실제로는 LLM 호출)
        user_input = self.state.user_input.lower()
        
        if "검색" in user_input or "찾" in user_input:
            plan = "1. 검색어 추출\n2. Google에서 검색\n3. 결과 분석\n4. 답변 제공"
        elif "이동" in user_input or "가" in user_input:
            plan = "1. 목표 URL 확인\n2. 페이지 이동\n3. 페이지 로딩 확인\n4. 결과 보고"
        else:
            plan = "1. 요청 분석\n2. 적절한 액션 결정\n3. 액션 실행\n4. 결과 확인"
        
        self.state.plan = plan
        print(f"   계획: {plan}")
    
    async def _thought_step(self):
        """Thought 단계: 다음 행동 결정"""
        print("🤔 Thought 단계: 다음 행동 결정 중...")
        
        user_input = self.state.user_input.lower()
        
        # 간단한 의사결정 로직 (실제로는 LLM 호출)
        if self.state.loop_count == 1:
            # 첫 번째 루프
            if "검색" in user_input or "찾" in user_input:
                # 검색어 추출
                search_query = self._extract_search_query(user_input)
                self.state.thought = f"'{search_query}'를 검색해야겠다."
                self.state.action = Action(
                    type=ActionType.SEARCH,
                    content=search_query
                )
            elif "이동" in user_input:
                # URL 추출 (간단한 예시)
                url = "https://www.google.com"  # 기본값
                self.state.thought = f"'{url}'로 이동해야겠다."
                self.state.action = Action(
                    type=ActionType.NAVIGATE,
                    target=url
                )
            else:
                # 기본 동작: 현재 페이지 정보 수집
                self.state.thought = "현재 페이지 정보를 확인해야겠다."
                self.state.action = Action(
                    type=ActionType.GET_DOM
                )
        else:
            # 후속 루프: 이전 결과에 따라 결정
            if not self.state.observation or "실패" in self.state.observation:
                self.state.thought = "이전 시도가 실패했으니 다른 방법을 시도해야겠다."
                self.state.action = Action(
                    type=ActionType.SCREENSHOT
                )
            else:
                self.state.thought = "충분한 정보를 얻었으니 완료해야겠다."
                self.state.action = None  # 액션 없음 (완료 신호)
        
        print(f"   사고: {self.state.thought}")
        if self.state.action:
            print(f"   계획된 액션: {self.state.action.type.value}")
    
    async def _action_step(self):
        """Action 단계: 실제 액션 실행"""
        print("⚡ Action 단계: 액션 실행 중...")
        
        if not self.state.action:
            self.state.observation = "실행할 액션이 없습니다."
            print("   실행할 액션이 없습니다.")
            return
        
        # 액션 실행
        result = await execute_action(self.state.action)
        
        if result["success"]:
            if self.state.action.type == ActionType.SEARCH:
                # 검색 결과에서 간단한 정보 추출
                text_content = await extract_text_content()
                self.state.observation = f"검색 완료. 결과: {text_content[:200] if text_content else '내용 없음'}..."
            elif self.state.action.type == ActionType.GET_DOM:
                # DOM 내용 요약
                content = result.get("data", "")
                self.state.observation = f"페이지 정보 수집 완료. 내용 길이: {len(content) if content else 0}자"
            else:
                self.state.observation = result["message"]
        else:
            self.state.observation = f"액션 실패: {result['message']}"
        
        print(f"   결과: {self.state.observation}")
    
    async def _explanation_step(self):
        """Explanation 단계: 결과 설명"""
        print("📝 Explanation 단계: 결과 해석 중...")
        
        if not self.state.observation:
            self.state.explanation = "관찰 결과가 없습니다."
        elif "실패" in self.state.observation or "오류" in self.state.observation:
            self.state.explanation = f"작업 중 문제가 발생했습니다: {self.state.observation}"
        elif self.state.action and self.state.action.type == ActionType.SEARCH:
            # 검색 결과 해석
            if "검색 완료" in self.state.observation:
                self.state.explanation = f"검색이 성공적으로 완료되었습니다. {self.state.observation}"
            else:
                self.state.explanation = "검색에 실패했습니다."
        else:
            self.state.explanation = f"작업이 완료되었습니다: {self.state.observation}"
        
        print(f"   설명: {self.state.explanation}")
    
    async def _critique_step(self):
        """Critique 단계: 완료 여부 판단"""
        print("🔍 Critique 단계: 완료 여부 판단 중...")
        
        # 완료 조건 확인
        done_conditions = [
            "검색 완료" in (self.state.observation or ""),
            "페이지 정보 수집 완료" in (self.state.observation or ""),
            self.state.loop_count >= 2,  # 최소 2번 시도 후 완료
            not self.state.action  # 액션이 없으면 완료
        ]
        
        if any(done_conditions):
            self.state.done = True
            print("   ✅ 작업 완료로 판단")
        else:
            self.state.done = False
            print("   🔄 추가 작업 필요")
        
        print(f"   완료 여부: {self.state.done}")
    
    def _extract_search_query(self, user_input: str) -> str:
        """사용자 입력에서 검색어 추출 (간단한 버전)"""
        # 간단한 키워드 추출
        keywords = ["검색", "찾", "알려줘", "뭐야", "어디", "언제", "누구"]
        
        for keyword in keywords:
            if keyword in user_input:
                # 키워드 이후 부분을 검색어로 사용
                parts = user_input.split(keyword)
                if len(parts) > 1:
                    query = parts[1].strip()
                    # 불필요한 문자 제거
                    query = query.replace("?", "").replace(".", "").replace("해줘", "").strip()
                    if query:
                        return query
        
        # 기본값: 전체 입력을 검색어로 사용
        return user_input.replace("?", "").strip()