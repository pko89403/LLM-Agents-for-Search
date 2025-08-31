"""
레스토랑 예약 전용 에이전트
OpenTable 등 예약 사이트에 특화된 기능
"""

import asyncio
from typing import Dict, Any, Optional
from agentq.playwright_helper import get_current_page, navigate_to
from agentq.web_selectors import OpenTableHelper, get_selector_for_site


class RestaurantReservationAgent:
    """레스토랑 예약 전용 에이전트"""
    
    def __init__(self):
        self.helper = OpenTableHelper()
    
    async def search_restaurants(
        self, 
        location: str = "New York", 
        restaurant: str = "",
        date: str = "",
        time: str = "",
        party_size: int = 2
    ) -> Dict[str, Any]:
        """레스토랑 검색"""
        
        try:
            # 1. OpenTable 접속
            print(f"🍽️ OpenTable에서 레스토랑 검색 중...")
            print(f"   위치: {location}")
            print(f"   레스토랑: {restaurant or '전체'}")
            print(f"   날짜: {date or '오늘'}")
            print(f"   시간: {time or '저녁'}")
            print(f"   인원: {party_size}명")
            
            success = await navigate_to("https://www.opentable.com")
            if not success:
                return {"success": False, "message": "OpenTable 접속 실패"}
            
            await asyncio.sleep(3)  # 페이지 로딩 대기
            
            # 2. 페이지 구조 분석
            page = await get_current_page()
            if not page:
                return {"success": False, "message": "페이지 접근 실패"}
            
            # 검색 입력 필드 찾기
            search_selectors = [
                'input[placeholder*="Location"]',
                'input[placeholder*="Restaurant"]', 
                'input[data-test*="search"]',
                'input[name*="search"]',
                '#search-input',
                '.search-input'
            ]
            
            search_input = None
            for selector in search_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        search_input = selector
                        break
                except:
                    continue
            
            if not search_input:
                # 페이지 내용 확인
                content = await page.evaluate("""
                    () => {
                        const inputs = document.querySelectorAll('input');
                        const buttons = document.querySelectorAll('button');
                        return {
                            inputs: Array.from(inputs).map(i => ({
                                type: i.type,
                                placeholder: i.placeholder,
                                name: i.name,
                                id: i.id
                            })),
                            buttons: Array.from(buttons).map(b => b.textContent?.trim()).slice(0, 10)
                        };
                    }
                """)
                
                return {
                    "success": False,
                    "message": "검색 입력 필드를 찾을 수 없음",
                    "page_info": content
                }
            
            # 3. 검색어 입력
            query = restaurant if restaurant else location
            await page.fill(search_input, query)
            print(f"   검색어 입력: {query}")
            
            # 4. 검색 실행 (Enter 키 또는 버튼 클릭)
            await page.press(search_input, 'Enter')
            await asyncio.sleep(3)
            
            # 5. 결과 확인
            current_url = page.url
            title = await page.title()
            
            # 레스토랑 목록 추출 시도
            restaurants = await page.evaluate("""
                () => {
                    const cards = document.querySelectorAll('[data-test*="restaurant"], .restaurant-card, .listing');
                    return Array.from(cards).slice(0, 5).map(card => {
                        const name = card.querySelector('h2, h3, .restaurant-name, .name')?.textContent?.trim();
                        const cuisine = card.querySelector('.cuisine, .category')?.textContent?.trim();
                        const rating = card.querySelector('.rating, .stars')?.textContent?.trim();
                        return { name, cuisine, rating };
                    }).filter(r => r.name);
                }
            """)
            
            return {
                "success": True,
                "message": f"검색 완료: {len(restaurants)}개 레스토랑 발견",
                "data": {
                    "query": query,
                    "url": current_url,
                    "title": title,
                    "restaurants": restaurants
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"검색 중 오류: {str(e)}"
            }
    
    async def get_reservation_info(self) -> Dict[str, Any]:
        """예약 정보 및 가이드 제공"""
        
        steps = self.helper.get_reservation_steps()
        issues = self.helper.get_common_issues()
        
        return {
            "success": True,
            "message": "OpenTable 예약 가이드",
            "data": {
                "steps": steps,
                "common_issues": issues,
                "tips": [
                    "인기 레스토랑은 미리 예약하세요",
                    "계정 생성을 미리 해두면 빠릅니다",
                    "전화 예약이 더 확실할 수 있습니다",
                    "취소 정책을 확인하세요"
                ]
            }
        }


# 편의 함수
async def search_opentable_restaurants(**kwargs) -> Dict[str, Any]:
    """OpenTable 레스토랑 검색 편의 함수"""
    agent = RestaurantReservationAgent()
    return await agent.search_restaurants(**kwargs)


async def get_opentable_guide() -> Dict[str, Any]:
    """OpenTable 예약 가이드 편의 함수"""
    agent = RestaurantReservationAgent()
    return await agent.get_reservation_info()