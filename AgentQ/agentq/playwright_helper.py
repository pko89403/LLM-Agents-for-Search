"""
Playwright 헬퍼 함수들
크롬 디버깅 모드에 직접 연결하는 간단한 유틸리티
"""

import asyncio
from typing import Optional
from playwright.async_api import async_playwright, Browser, Page, Playwright, BrowserContext


# 전역 변수로 간단하게 관리
_playwright: Optional[Playwright] = None
_browser: Optional[Browser] = None
_page: Optional[Page] = None


class PlaywrightHelper:
    """Playwright 헬퍼 클래스 - 테스트 시스템과 통합용"""

    def __init__(self):
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.headless: bool = True

    async def setup(self, headless: bool = True, debug_port: Optional[int] = None):
        """브라우저 설정 및 시작"""
        self.headless = headless

        try:
            # Playwright 시작
            self.playwright = await async_playwright().start()

            if debug_port:
                # 디버깅 모드 Chrome에 연결
                self.browser = await self.playwright.chromium.connect_over_cdp(
                    f"http://localhost:{debug_port}"
                )
                contexts = self.browser.contexts
                if contexts and contexts[0].pages:
                    self.page = contexts[0].pages[0]
                else:
                    self.context = await self.browser.new_context()
                    self.page = await self.context.new_page()
            else:
                # 새 브라우저 인스턴스 시작
                self.browser = await self.playwright.chromium.launch(
                    headless=headless,
                    args=['--no-sandbox', '--disable-dev-shm-usage']
                )
                self.context = await self.browser.new_context()
                self.page = await self.context.new_page()

            print(f"✅ 브라우저 설정 완료 (headless: {headless})")

        except Exception as e:
            print(f"❌ 브라우저 설정 실패: {e}")
            raise

    async def cleanup(self):
        """리소스 정리"""
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()

            print("✅ Playwright 리소스 정리 완료")

        except Exception as e:
            print(f"❌ 리소스 정리 중 오류: {e}")

    async def navigate_to(self, url: str) -> bool:
        """지정된 URL로 이동"""
        try:
            if not self.page:
                return False

            await self.page.goto(url, wait_until="load", timeout=30000)
            print(f"📍 이동: {url}")
            return True

        except Exception as e:
            print(f"❌ 페이지 이동 실패: {e}")
            return False

    async def click_element(self, selector: str) -> bool:
        """요소 클릭"""
        try:
            if not self.page:
                return False

            await self.page.click(selector, timeout=10000)
            print(f"🖱️ 클릭: {selector}")
            return True

        except Exception as e:
            print(f"❌ 클릭 실패: {e}")
            return False

    async def type_text(self, selector: str, text: str) -> bool:
        """텍스트 입력"""
        try:
            if not self.page:
                return False

            await self.page.fill(selector, text)
            print(f"⌨️ 입력: {text} → {selector}")
            return True

        except Exception as e:
            print(f"❌ 텍스트 입력 실패: {e}")
            return False

    async def get_page_title(self) -> Optional[str]:
        """페이지 제목 가져오기"""
        try:
            if not self.page:
                return None

            return await self.page.title()

        except Exception as e:
            print(f"❌ 제목 가져오기 실패: {e}")
            return None

    async def get_page_url(self) -> Optional[str]:
        """현재 페이지 URL 가져오기"""
        try:
            if not self.page:
                return None

            return self.page.url

        except Exception as e:
            print(f"❌ URL 가져오기 실패: {e}")
            return None

    async def take_screenshot(self, path: str = "screenshot.png") -> bool:
        """스크린샷 촬영"""
        try:
            if not self.page:
                return False

            await self.page.screenshot(path=path)
            print(f"📸 스크린샷 저장: {path}")
            return True

        except Exception as e:
            print(f"❌ 스크린샷 실패: {e}")
            return False

    async def get_page_content(self) -> Optional[str]:
        """페이지 HTML 내용 가져오기"""
        try:
            if not self.page:
                return None

            return await self.page.content()

        except Exception as e:
            print(f"❌ 페이지 내용 가져오기 실패: {e}")
            return None


async def connect_to_chrome(debug_port: int = 9222) -> Optional[Page]:
    """
    실행 중인 Chrome 디버깅 모드에 연결

    Args:
        debug_port: Chrome 디버깅 포트 (기본값: 9222)

    Returns:
        Page 객체 또는 None
    """
    global _playwright, _browser, _page

    try:
        # Playwright 시작
        if not _playwright:
            _playwright = await async_playwright().start()

        # Chrome에 연결
        if not _browser:
            _browser = await _playwright.chromium.connect_over_cdp(
                f"http://localhost:{debug_port}"
            )

        # 페이지 가져오기 또는 생성
        if not _page:
            contexts = _browser.contexts
            if contexts and contexts[0].pages:
                _page = contexts[0].pages[0]
            else:
                context = await _browser.new_context()
                _page = await context.new_page()

        print(f"✅ Chrome에 연결됨 (포트: {debug_port})")
        return _page

    except Exception as e:
        print(f"❌ Chrome 연결 실패: {e}")
        return None


async def get_current_page() -> Optional[Page]:
    """현재 페이지 반환 (연결되지 않았으면 자동 연결 시도)"""
    global _page

    if _page:
        return _page

    # 자동 연결 시도
    return await connect_to_chrome()


async def navigate_to(url: str) -> bool:
    """지정된 URL로 이동"""
    try:
        page = await get_current_page()
        if not page:
            return False

        await page.goto(url)
        print(f"📍 이동: {url}")
        return True

    except Exception as e:
        print(f"❌ 페이지 이동 실패: {e}")
        return False


async def take_screenshot(path: str = "screenshot.png") -> bool:
    """스크린샷 촬영"""
    try:
        page = await get_current_page()
        if not page:
            return False

        await page.screenshot(path=path)
        print(f"📸 스크린샷 저장: {path}")
        return True

    except Exception as e:
        print(f"❌ 스크린샷 실패: {e}")
        return False


async def get_page_title() -> Optional[str]:
    """페이지 제목 가져오기"""
    try:
        page = await get_current_page()
        if not page:
            return None

        return await page.title()

    except Exception as e:
        print(f"❌ 제목 가져오기 실패: {e}")
        return None


async def get_page_url() -> Optional[str]:
    """현재 페이지 URL 가져오기"""
    try:
        page = await get_current_page()
        if not page:
            return None

        return page.url

    except Exception as e:
        print(f"❌ URL 가져오기 실패: {e}")
        return None


async def cleanup():
    """리소스 정리"""
    global _playwright, _browser, _page

    try:
        if _page:
            await _page.close()
            _page = None

        if _browser:
            await _browser.close()
            _browser = None

        if _playwright:
            await _playwright.stop()
            _playwright = None

        print("✅ Playwright 리소스 정리 완료")

    except Exception as e:
        print(f"❌ 리소스 정리 중 오류: {e}")


# 편의 함수들
async def click_element(selector: str) -> bool:
    """요소 클릭"""
    try:
        page = await get_current_page()
        if not page:
            return False

        await page.click(selector)
        print(f"🖱️ 클릭: {selector}")
        return True

    except Exception as e:
        print(f"❌ 클릭 실패: {e}")
        return False


async def type_text(selector: str, text: str) -> bool:
    """텍스트 입력"""
    try:
        page = await get_current_page()
        if not page:
            return False

        await page.fill(selector, text)
        print(f"⌨️ 입력: {text} → {selector}")
        return True

    except Exception as e:
        print(f"❌ 텍스트 입력 실패: {e}")
        return False


async def get_page_content() -> Optional[str]:
    """페이지 HTML 내용 가져오기"""
    try:
        page = await get_current_page()
        if not page:
            return None

        return await page.content()

    except Exception as e:
        print(f"❌ 페이지 내용 가져오기 실패: {e}")
        return None

async def index_interactive_elements():
    try:
        page = await get_current_page()
        if not page:
            return []
        elements = await page.evaluate("""
            () => {
                function visible(el){
                    const r = el.getBoundingClientRect();
                    const cs = getComputedStyle(el);
                    return r.width > 0 && r.height > 0 && cs.visibility !== 'hidden' && cs.display !== 'none';
                }
                const nodes = Array.from(document.querySelectorAll(
                  'button, a[href], input, select, textarea, [role="button"]'
                )).filter(visible);
                let i = 0;
                nodes.forEach(el => { if(!el.dataset.agentqId){ el.dataset.agentqId = 'el_' + (++i); }});
                return nodes.slice(0,200).map(el => ({
                    id: el.dataset.agentqId,
                    dom_id: el.id || '',
                    tag: el.tagName.toLowerCase(),
                    role: el.getAttribute('role') || (el.tagName.toLowerCase()==='a' ? 'link' :
                         (el.tagName.toLowerCase()==='button' ? 'button' :
                         (['input','select','textarea'].includes(el.tagName.toLowerCase())?'input':''))),
                    text: (el.innerText||'').trim().slice(0,80),
                    placeholder: el.getAttribute('placeholder') || '',
                    type: el.getAttribute('type') || '',
                    name: el.getAttribute('name') || '',
                    href: el.getAttribute('href') || ''
                }));
            }
        """)
        return elements or []
    except Exception as e:
        print(f"❌ index_interactive_elements 오류: {e}")
        return []

async def click_by_agentq_id(agentq_id: str) -> bool:
    try:
        page = await get_current_page()
        if not page:
            return False

        loc = page.locator(f'[data-agentq-id="{agentq_id}"]')
        if await loc.count() == 0:
            if agentq_id.startswith('#'):
                loc = page.locator(agentq_id)
            else:
                loc = page.locator(f'[id="{agentq_id}"]')
                if await loc.count() == 0:
                    loc = page.locator(agentq_id)

        if await loc.count() == 0:
            return False

        await loc.first.click()
        print(f"🖱️ 클릭(ID/by): {agentq_id}")
        return True
    except Exception as e:
        print(f"❌ click_by_agentq_id 오류: {e}")
        return False

async def set_input_by_agentq_id(agentq_id: str, text: str) -> bool:
    try:
        page = await get_current_page()
        if not page:
            return False

        # Prefer data-agentq-id
        loc = page.locator(f'[data-agentq-id="{agentq_id}"]')
        if await loc.count() == 0:
            # If starts with '#', assume it's a CSS selector as-is (could be an id)
            if agentq_id.startswith('#'):
                loc = page.locator(agentq_id)
            else:
                # Try native DOM id (without CSS escaping issues)
                loc = page.locator(f'[id="{agentq_id}"]')
                if await loc.count() == 0:
                    # Treat the whole string as a CSS selector fallback
                    loc = page.locator(agentq_id)

        if await loc.count() == 0:
            return False

        # Attempt direct fill (works for <input>, <textarea>, contenteditable in Playwright >=1.41)
        try:
            await loc.fill(text)
        except Exception:
            # Try click, select-all, and type (some input masks need focus)
            try:
                await loc.click(force=True)
            except Exception:
                pass
            try:
                await loc.press('Control+A')
                await loc.type(text)
            except Exception:
                # Final fallback for contentEditable or non-standard widgets
                try:
                    await loc.evaluate('(el, v) => { if ("value" in el) el.value = v; else el.innerText = v; el.dispatchEvent(new Event("input", {bubbles:true})); el.dispatchEvent(new Event("change", {bubbles:true})); }', text)
                except Exception as e:
                    print(f"❌ set_input_by_agentq_id 최종 폴백 실패: {e}")
                    return False

        print(f"⌨️ 입력(ID/by): {agentq_id} ← {text}")
        return True
    except Exception as e:
        print(f"❌ set_input_by_agentq_id 오류: {e}")
        return False

async def clear_by_agentq_id(agentq_id: str) -> bool:
    return await set_input_by_agentq_id(agentq_id, "")

async def submit_by_agentq_id(agentq_id: str) -> bool:
    try:
        page = await get_current_page()
        if not page:
            return False

        loc = page.locator(f'[data-agentq-id="{agentq_id}"]')
        if await loc.count() == 0:
            if agentq_id.startswith('#'):
                loc = page.locator(agentq_id)
            else:
                loc = page.locator(f'[id="{agentq_id}"]')
                if await loc.count() == 0:
                    loc = page.locator(agentq_id)

        if await loc.count() == 0:
            return False

        # Prefer form submission if available
        try:
            await loc.evaluate('(el) => { if (el.form) { el.form.requestSubmit ? el.form.requestSubmit() : el.form.submit(); } else { el.click(); } }')
        except Exception:
            await loc.click()

        print(f"📨 제출(ID/by): {agentq_id}")
        return True
    except Exception as e:
        print(f"❌ submit_by_agentq_id 오류: {e}")
        return False

async def get_dom_snapshot():
    try:
        page = await get_current_page()
        if not page: return None
        elements = await index_interactive_elements()
        title = await page.title()
        url = page.url
        text = await page.evaluate("() => document.body.innerText.slice(0, 3000)")
        return {"title": title, "url": url, "content": text, "elements": elements}
    except Exception as e:
        print(f"❌ get_dom_snapshot 오류: {e}")
        return None


async def find_and_use_search_bar(query: str) -> bool:
    """페이지 내에서 검색창을 찾아 검색을 시도"""
    try:
        page = await get_current_page()
        if not page:
            return False

        elements = await index_interactive_elements()
        if not elements:
            return False

        print("--- 페이지 내 상호작용 요소 분석 ---")
        for i, el in enumerate(elements[:15]): # 처음 15개 요소만 로그로 출력
            print(f"  - Element {i}: id={el.get('id')}, role={el.get('role')}, placeholder={el.get('placeholder')}, name={el.get('name')}, text={el.get('text')}")
        print("------------------------------------")

        search_input = None
        # 다양한 단서를 기반으로 검색창 찾기
        for el in elements:
            el_id = el.get('id', '')
            el_role = el.get('role', '')
            el_placeholder = el.get('placeholder', '').lower()
            el_name = el.get('name', '').lower()
            el_text = el.get('text', '').lower()

            if 'search' in el_role or 'searchbox' in el_role:
                search_input = el
                break
            if 'search' in el_placeholder or '검색' in el_placeholder:
                search_input = el
                break
            if el_name in ['q', 's', 'query', 'search']:
                search_input = el
                break
        
        if not search_input:
            print("   페이지 내에서 검색창을 찾지 못했습니다.")
            return False

        input_id = search_input['id']
        print(f"   검색창 찾음 (ID: {input_id}). 텍스트 입력 시도...")
        success = await set_input_by_agentq_id(input_id, query)
        if not success:
            print(f"   검색창(ID: {input_id})에 텍스트 입력 실패.")
            return False

        # 검색 버튼 찾기 (입력창 주변)
        submit_button = None
        for el in elements:
            el_type = el.get('type', '').lower()
            el_role = el.get('role', '')
            el_text = el.get('text', '').lower()

            if el_type == 'submit':
                submit_button = el
                break
            if 'search' in el_text or '검색' in el_text:
                 if el.get('tag') == 'button' or 'button' in el_role:
                    submit_button = el
                    break

        if submit_button:
            button_id = submit_button['id']
            print(f"   검색 버튼 찾음 (ID: {button_id}). 클릭 시도...")
            await click_by_agentq_id(button_id)
        else:
            # 버튼이 없으면 Enter 키 입력 시도
            print("   검색 버튼을 찾지 못했습니다. Enter 키 입력을 시도합니다.")
            await page.locator(f'[data-agentq-id="{input_id}"]').press('Enter')
        
        await page.wait_for_load_state('load', timeout=15000)
        print(f"   페이지 내 검색 성공: {query}")
        return True

    except Exception as e:
        print(f"❌ find_and_use_search_bar 오류: {e}")
        return False
