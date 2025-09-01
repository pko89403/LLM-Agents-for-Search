"""
에이전트가 사용하는 외부 도구 구현
"""

import asyncio
from typing import Dict, Any, Optional, List
from agentq.playwright_helper import (
    get_current_page, navigate_to, take_screenshot,
    click_element, type_text, get_page_content,
    get_page_title, get_page_url,
    index_interactive_elements, get_dom_snapshot,
    click_by_agentq_id, set_input_by_agentq_id, submit_by_agentq_id, clear_by_agentq_id,
    find_and_use_search_bar
)


class WebTool:
    """웹 상호작용 도구"""
    
    @staticmethod
    async def navigate(url: str) -> Dict[str, Any]:
        """페이지 이동"""
        try:
            success = await navigate_to(url)
            if success:
                title = await get_page_title()
                current_url = await get_page_url()
                return {
                    "success": True,
                    "message": f"페이지 이동 성공: {url}",
                    "data": {
                        "url": current_url,
                        "title": title
                    }
                }
            else:
                return {
                    "success": False,
                    "message": f"페이지 이동 실패: {url}",
                    "data": None
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"페이지 이동 중 오류: {str(e)}",
                "data": None
            }
    
    @staticmethod
    async def search(query: str) -> Dict[str, Any]:
        """현재 페이지에서 검색을 시도하고, 실패하면 Google 검색으로 대체"""
        try:
            # 1. 현재 페이지에서 검색 시도
            on_site_success = await find_and_use_search_bar(query)
            if on_site_success:
                return {
                    "success": True,
                    "message": f"페이지 내 검색 완료: {query}",
                    "data": None
                }

            # 2. 실패 시 Google 검색으로 대체
            print("   페이지 내 검색 실패. Google 검색으로 대체합니다.")
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            success = await navigate_to(search_url)
            
            if success:
                await asyncio.sleep(2)  # 페이지 로딩 대기
                page = await get_current_page()
                if page:
                    search_results = await page.evaluate("""
                        () => {
                            const results = [];
                            const resultElements = document.querySelectorAll('div[data-ved] h3, .g h3, .rc h3');
                            const descElements = document.querySelectorAll('.VwiC3b, .s3v9rd, .st');
                            for (let i = 0; i < Math.min(5, resultElements.length); i++) {
                                const title = resultElements[i]?.innerText || '';
                                const desc = descElements[i]?.innerText || '';
                                if (title) {
                                    results.push(`${title}: ${desc}`);
                                }
                            }
                            return results.join('\n\n');
                        }
                    """)
                    return {
                        "success": True,
                        "message": f"Google 검색 완료: {query}",
                        "data": search_results
                    }
            
            return {
                "success": False,
                "message": f"Google 검색 페이지 이동 실패: {search_url}",
                "data": None
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"검색 중 오류: {str(e)}",
                "data": None
            }
    
    @staticmethod
    async def click(selector: str) -> Dict[str, Any]:
        """요소 클릭"""
        try:
            success = await click_element(selector)
            return {
                "success": success,
                "message": f"클릭 {'성공' if success else '실패'}: {selector}",
                "data": None
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"클릭 중 오류: {str(e)}",
                "data": None
            }
    
    @staticmethod
    async def type_text_input(selector: str, text: str) -> Dict[str, Any]:
        """텍스트 입력"""
        try:
            success = await type_text(selector, text)
            return {
                "success": success,
                "message": f"텍스트 입력 {'성공' if success else '실패'}: {text}",
                "data": None
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"텍스트 입력 중 오류: {str(e)}",
                "data": None
            }
    
    @staticmethod
    async def capture_screenshot(path: str = "screenshot.png") -> Dict[str, Any]:
        """스크린샷 촬영"""
        try:
            success = await take_screenshot(path)
            return {
                "success": success,
                "message": f"스크린샷 {'저장' if success else '실패'}: {path}",
                "data": path if success else None
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"스크린샷 중 오류: {str(e)}",
                "data": None
            }
    
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
    
    @staticmethod
    async def wait(seconds: int) -> Dict[str, Any]:
        """대기"""
        try:
            await asyncio.sleep(seconds)
            return {
                "success": True,
                "message": f"{seconds}초 대기 완료",
                "data": None
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"대기 중 오류: {str(e)}",
                "data": None
            }
    
    @staticmethod
    async def scroll_page(direction: str = "down", amount: int = 3) -> Dict[str, Any]:
        """페이지 스크롤"""
        try:
            page = await get_current_page()
            if not page:
                return {
                    "success": False,
                    "message": "페이지에 접근할 수 없습니다",
                    "data": None
                }
            
            if direction.lower() == "down":
                await page.evaluate(f"window.scrollBy(0, {amount * 300})")
            elif direction.lower() == "up":
                await page.evaluate(f"window.scrollBy(0, {-amount * 300})")
            else:
                return {
                    "success": False,
                    "message": f"지원하지 않는 스크롤 방향: {direction}",
                    "data": None
                }
            
            return {
                "success": True,
                "message": f"페이지 스크롤 완료: {direction}",
                "data": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"스크롤 중 오류: {str(e)}",
                "data": None
            }


class ToolExecutor:
    """도구 실행기"""
    
    def __init__(self):
        self.web_tool = WebTool()
    
    async def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """액션 실행"""
        action_type = action.get("type", "").upper()
        
        try:
            if action_type == "NAVIGATE":
                return await self.web_tool.navigate(action.get("target", ""))
            
            elif action_type == "SEARCH":
                return await self.web_tool.search(action.get("content", ""))
            
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
            
            elif action_type == "SCREENSHOT":
                path = action.get("target", "screenshot.png")
                return await self.web_tool.capture_screenshot(path)
            
            elif action_type == "GET_DOM":
                return await self.web_tool.extract_page_content()
            
            elif action_type == "WAIT":
                seconds = int(action.get("content", "1"))
                return await self.web_tool.wait(seconds)
            
            elif action_type == "SCROLL":
                direction = action.get("target", "down")
                return await self.web_tool.scroll_page(direction)
            
            else:
                return {
                    "success": False,
                    "message": f"지원하지 않는 액션 타입: {action_type}",
                    "data": None
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"액션 실행 중 오류: {str(e)}",
                "data": None
            }


# 전역 도구 실행기 인스턴스
_tool_executor: Optional[ToolExecutor] = None
_playwright_helper = None  # 테스트 시스템과 통합용

def set_playwright_helper(helper):
    """PlaywrightHelper 인스턴스 설정 (테스트 시스템 통합용)"""
    global _playwright_helper
    _playwright_helper = helper

def get_playwright_helper():
    """현재 설정된 PlaywrightHelper 반환"""
    return _playwright_helper

def get_tool_executor() -> ToolExecutor:
    """도구 실행기 싱글톤 인스턴스 반환"""
    global _tool_executor
    if _tool_executor is None:
        _tool_executor = ToolExecutor()
    return _tool_executor