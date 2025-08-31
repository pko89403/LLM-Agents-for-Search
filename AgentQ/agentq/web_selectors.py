"""
웹사이트별 CSS 선택자 및 상호작용 패턴 정의
"""

from typing import Dict, List, Optional


class WebSiteSelectors:
    """웹사이트별 선택자 모음"""
    
    OPENTABLE = {
        "search_input": 'input[data-test="typeahead-input"]',
        "search_button": 'button[data-test="submit-search"]',
        "date_picker": 'button[data-test="date-picker"]',
        "time_select": 'select[data-test="time-select"]',
        "party_size": 'select[data-test="party-size-select"]',
        "restaurant_card": '[data-test="restaurant-card"]',
        "book_button": 'button[data-test="book-button"]',
        "login_button": 'button[data-test="login-button"]',
        "signup_button": 'button[data-test="signup-button"]'
    }
    
    GOOGLE = {
        "search_input": 'input[name="q"]',
        "search_button": 'input[name="btnK"]',
        "results": '#search .g'
    }
    
    COMMON = {
        "submit_button": 'input[type="submit"], button[type="submit"]',
        "search_input": 'input[type="search"], input[name*="search"], input[placeholder*="search"]',
        "email_input": 'input[type="email"], input[name*="email"]',
        "password_input": 'input[type="password"], input[name*="password"]'
    }


class OpenTableHelper:
    """OpenTable 전용 헬퍼 클래스"""
    
    @staticmethod
    def get_search_strategy(location: str = "", restaurant: str = "") -> Dict:
        """검색 전략 반환"""
        if restaurant:
            return {
                "type": "restaurant_search",
                "query": restaurant,
                "location": location
            }
        else:
            return {
                "type": "location_search", 
                "query": location or "New York"
            }
    
    @staticmethod
    def get_reservation_steps() -> List[str]:
        """예약 단계 반환"""
        return [
            "1. OpenTable 웹사이트 접속",
            "2. 위치 또는 레스토랑 검색",
            "3. 날짜 선택",
            "4. 시간 선택", 
            "5. 인원수 선택",
            "6. 레스토랑 선택",
            "7. 예약 확인 (로그인 필요할 수 있음)"
        ]
    
    @staticmethod
    def get_common_issues() -> List[str]:
        """일반적인 문제점들"""
        return [
            "로그인이 필요할 수 있음",
            "CAPTCHA 인증이 필요할 수 있음",
            "신용카드 정보가 필요할 수 있음",
            "특정 레스토랑은 전화 예약만 가능",
            "시간대별 예약 가능 여부가 다름"
        ]


def get_selector_for_site(site: str, element: str) -> Optional[str]:
    """사이트별 선택자 반환"""
    selectors = getattr(WebSiteSelectors, site.upper(), {})
    return selectors.get(element)