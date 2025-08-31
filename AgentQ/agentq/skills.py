"""
AgentQ 웹 스킬 구현
기본적인 웹 상호작용 기능들
"""

import asyncio
from typing import Optional, Dict, Any
from agentq.playwright_helper import (
    get_current_page, navigate_to, take_screenshot,
    click_element, type_text, get_page_content,
    get_page_title, get_page_url
)
from agentq.models import Action, ActionType


async def execute_action(action: Action) -> Dict[str, Any]:
    """
    액션 실행
    
    Args:
        action: 실행할 액션
    
    Returns:
        실행 결과 딕셔너리
    """
    result = {
        "success": False,
        "message": "",
        "data": None
    }
    
    try:
        if action.type == ActionType.NAVIGATE:
            success = await navigate_to(action.target)
            result["success"] = success
            result["message"] = f"페이지 이동: {action.target}" if success else "페이지 이동 실패"
            
        elif action.type == ActionType.CLICK:
            success = await click_element(action.target)
            result["success"] = success
            result["message"] = f"클릭: {action.target}" if success else "클릭 실패"
            
        elif action.type == ActionType.TYPE:
            success = await type_text(action.target, action.content)
            result["success"] = success
            result["message"] = f"텍스트 입력: {action.content}" if success else "텍스트 입력 실패"
            
        elif action.type == ActionType.SCREENSHOT:
            path = action.target or "screenshot.png"
            success = await take_screenshot(path)
            result["success"] = success
            result["message"] = f"스크린샷 저장: {path}" if success else "스크린샷 실패"
            result["data"] = path if success else None
            
        elif action.type == ActionType.GET_DOM:
            content = await get_page_content()
            result["success"] = content is not None
            result["message"] = "DOM 추출 완료" if content else "DOM 추출 실패"
            result["data"] = content
            
        elif action.type == ActionType.SEARCH:
            # 간단한 Google 검색 구현
            search_url = f"https://www.google.com/search?q={action.content}"
            success = await navigate_to(search_url)
            if success:
                await asyncio.sleep(2)  # 페이지 로딩 대기
                content = await get_page_content()
                result["success"] = True
                result["message"] = f"검색 완료: {action.content}"
                result["data"] = content
            else:
                result["message"] = "검색 실패"
                
        elif action.type == ActionType.WAIT:
            wait_time = int(action.content) if action.content else 1
            await asyncio.sleep(wait_time)
            result["success"] = True
            result["message"] = f"{wait_time}초 대기 완료"
            
        else:
            result["message"] = f"지원하지 않는 액션 타입: {action.type}"
            
    except Exception as e:
        result["message"] = f"액션 실행 중 오류: {str(e)}"
    
    return result


async def get_page_info() -> Dict[str, Any]:
    """현재 페이지 정보 수집"""
    try:
        title = await get_page_title()
        url = await get_page_url()
        
        return {
            "title": title,
            "url": url,
            "success": True
        }
    except Exception as e:
        return {
            "title": None,
            "url": None,
            "success": False,
            "error": str(e)
        }


async def extract_text_content() -> Optional[str]:
    """페이지에서 텍스트 내용만 추출 (간단한 버전)"""
    try:
        page = await get_current_page()
        if not page:
            return None
        
        # 주요 텍스트 요소들에서 내용 추출
        text_content = await page.evaluate("""
            () => {
                // 불필요한 요소들 제거
                const elementsToRemove = ['script', 'style', 'nav', 'header', 'footer'];
                elementsToRemove.forEach(tag => {
                    const elements = document.getElementsByTagName(tag);
                    for (let i = elements.length - 1; i >= 0; i--) {
                        elements[i].remove();
                    }
                });
                
                // 주요 텍스트 내용 추출
                const textElements = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'div', 'a'];
                let text = '';
                
                textElements.forEach(tag => {
                    const elements = document.getElementsByTagName(tag);
                    for (let element of elements) {
                        const elementText = element.innerText?.trim();
                        if (elementText && elementText.length > 10) {
                            text += elementText + '\\n';
                        }
                    }
                });
                
                return text.slice(0, 2000); // 처음 2000자만 반환
            }
        """)
        
        return text_content
        
    except Exception as e:
        print(f"텍스트 추출 중 오류: {e}")
        return None