'''
WebShop 환경과 상호작용하기 위한 액션(Action)과 도구(Tool)를 정의합니다.
'search'와 'choose' 두 가지 고수준 액션으로 단순화된 최종 버전입니다.
'''
import os
import re
import requests
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from bs4 import BeautifulSoup

# state.py에서 정의한 Observation과 Product를 가져옵니다.
from state import Observation, Product

# --- 액션 정의 ---

class ActType(str, Enum):
    """WebShop에서 수행 가능한 액션 유형"""
    SEARCH = "search"
    CHOOSE = "choose"

@dataclass(frozen=True)
class Action:
    """WebShop 액션 데이터 모델"""
    type: ActType
    parameter: str  # search의 경우 query, choose의 경우 클릭할 버튼/링크의 텍스트

    def __str__(self):
        """액션을 문자열로 표현"""
        return f"{self.type.value}('{self.parameter}')"

# --- WebShop 클라이언트 ---

class WebShopClient:
    """
    WebShop 시뮬레이션 환경과 상호작용하는 클라이언트.
    'search[query]'와 'choose[button_text]' 액션을 처리합니다.
    """
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.environ.get("WEBSHOP_BASE_URL", "http://localhost:3000")
        self.session = requests.Session()

    def _parse_html_observation(self, html_content: str, query: Optional[str] = None) -> Observation:
        """
        HTML 응답을 Observation 객체로 파싱합니다.
        성공한 테스트 스크립트의 파싱 로직을 기반으로 합니다.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 1. 상품 정보 파싱
        products = []
        product_containers = soup.find_all('div', class_='col-lg-12 mx-auto list-group-item')
        for i, container in enumerate(product_containers[:10]):
            try:
                asin_elem = container.find('h4', class_='product-asin')
                asin_link = asin_elem.find('a', class_='product-link') if asin_elem else None
                product_id = asin_link.get_text(strip=True) if asin_link else f"item_{i}"
                
                title_elem = container.find('h4', class_='product-title')
                title = title_elem.get_text(strip=True) if title_elem else f"상품 {i+1}"
                
                price_elem = container.find('h5', class_='product-price')
                price = 0.0
                if price_elem:
                    price_text = price_elem.get_text(strip=True)
                    # 가격 범위 처리 (e.g., "$6.63 to $8.37" -> 6.63)
                    price_match = re.search(r'\$(\d+\.?\d*)', price_text)
                    if price_match:
                        price = float(price_match.group(1))
                
                products.append(Product(id=product_id, title=title, price=price))
            except Exception:
                continue # 개별 상품 파싱 실패시 건너뛰기

        # 2. 클릭 가능한 액션 파싱 (텍스트 -> URL/액션 맵)
        clickable_elements: Dict[str, str] = {}

        # 상품 링크 (ID)
        for p in products:
            # Assuming product IDs link to /item/{id}
            clickable_elements[p.id] = f"/item/{p.id}"

        # 버튼 및 링크
        # Find all buttons and links that are likely clickable actions
        for element in soup.find_all(['button', 'a'], class_=['btn', 'product-link']):
            text = element.get_text(strip=True)
            if text:
                if element.name == 'a' and element.get('href'):
                    # For links, store the href
                    clickable_elements[text] = element.get('href')
                elif element.name == 'button':
                    # For buttons, we'll use a placeholder for now.
                    # Real implementation would need to handle form submissions or JS clicks.
                    clickable_elements[text] = 'BUTTON_CLICK' # Placeholder

        # 3. 검색창 존재 여부 확인
        has_search_bar = soup.find('input', {'name': 'search_query'}) is not None

        return Observation(
            url="", # self.session.get_redirect_history()는 response 객체에서만 사용 가능하여 임시 처리
            query=query,
            page=1, # TODO: 페이지 정보 파싱 필요
            sort=None, # TODO: 정렬 정보 파싱 필요
            filters={}, # TODO: 필터 정보 파싱 필요
            results=products,
            cart=[], # TODO: 장바구니 정보 파싱 필요
            available_actions={'has_search_bar': has_search_bar, 'clickables': clickable_elements},
            html=html_content
        )

    def reset(self) -> Observation:
        """환경을 초기화하고 초기 관찰값을 반환합니다."""
        try:
            response = self.session.get(f"{self.base_url}/", timeout=10)
            response.raise_for_status()
            return self._parse_html_observation(response.text)
        except requests.RequestException as e:
            print(f"Error during reset: {e}")
            return Observation(url=None, query=None, page=1, sort=None, filters={}, results=[], cart=[], available_actions={'has_search_bar': False, 'clickables': []}, html=str(e))

    def search(self, query: str) -> Observation:
        """'search[query]' 액션을 실행합니다."""
        try:
            response = self.session.post(
                f"{self.base_url}/abc", # WebShop의 검색 URL 경로
                data={'search_query': query},
                timeout=10,
                allow_redirects=True
            )
            response.raise_for_status()
            return self._parse_html_observation(response.text, query=query)
        except requests.RequestException as e:
            print(f"Error during search: {e}")
            return Observation(url=None, query=query, page=1, sort=None, filters={}, results=[], cart=[], available_actions={'has_search_bar': False, 'clickables': []}, html=str(e))

    def choose(self, target: str, current_observation: Observation) -> Observation:
        """'choose[button_text]' 액션을 실행합니다."""
        try:
            clickable_map = current_observation.available_actions.get('clickables', {})
            action_url = clickable_map.get(target)

            if action_url:
                if action_url == 'BUTTON_CLICK':
                    # This is a placeholder for buttons that don't have a direct URL (e.g., form submit buttons)
                    # For now, we'll assume it's a simple click that might refresh the page or trigger JS.
                    # A robust solution would require finding the form and submitting it.
                    # For now, we'll just go to the current URL as a fallback.
                    response = self.session.get(current_observation.url or f"{self.base_url}/", timeout=10)
                elif action_url.startswith('/'): # Relative URL
                    response = self.session.get(f"{self.base_url}{action_url}", timeout=10)
                else: # Absolute URL
                    response = self.session.get(action_url, timeout=10)
            else:
                # Fallback if the target is not found in clickable_map
                # This might happen if LLM hallucinates an action or if it's a complex form submission.
                # For now, go to the home page.
                print(f"경고: '{target}'에 해당하는 클릭 가능한 액션을 찾을 수 없습니다. 홈 페이지로 이동합니다.")
                response = self.session.get(f"{self.base_url}/", timeout=10)

            response.raise_for_status()
            return self._parse_html_observation(response.text)
        except requests.RequestException as e:
            print(f"Error during choose: {e}")
            return Observation(url=None, query=None, page=1, sort=None, filters={}, results=[], cart=[], available_actions={'has_search_bar': False, 'clickables': {}}, html=str(e))


# --- 상태 전이 함수 ---

webshop_client = WebShopClient()

def transition(state: Dict[str, Any], action: Action) -> Dict[str, Any]:
    """
    주어진 상태(state)에서 액션(action)을 수행하고 다음 상태를 반환합니다.
    """
    if action.type == ActType.SEARCH:
        new_observation = webshop_client.search(action.parameter)
    elif action.type == ActType.CHOOSE:
        new_observation = webshop_client.choose(action.parameter, state['observation'])
    else:
        raise ValueError(f"Unsupported action type: {action.type}")

    return {
        "observation": new_observation,
        "action_history": state.get("action_history", []) + [str(action)],
    }
