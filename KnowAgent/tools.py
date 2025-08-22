from __future__ import annotations

import os
from typing import List
import re
import requests
from urllib.parse import quote as _urlquote
from html import unescape as _html_unescape

from langchain_community.docstore.wikipedia import Wikipedia

WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
WIKI_OPENSEARCH_URL = "https://en.wikipedia.org/w/api.php"
BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"

def wikipedia_retrieve(entity: str) -> str:
    """위키피디아에서 엔티티의 요약을 가져옵니다. 실패 시 유사 엔티티를 반환합니다.

    우선순위:
    1) langchain_community Wikipedia(docstore) 설치 시: 검색 + 문단 추출
    2) REST summary API
    3) OpenSearch 폴백
    """
    entity = (entity or "").strip()
    if not entity:
        return "No entity provided."

    # 1) langchain_community Wikipedia 사용 시도
    try:
        wiki = Wikipedia()
        # The docstore returns a list of Documents with .pagecontent
        docs = wiki.search(entity)
        if docs:
            # 간단히 첫 문서 내용의 첫 3~4문장 정도 반환
            text = (
                docs[0].pagecontent
                if hasattr(docs[0], "pagecontent")
                else str(docs[0])
            )
            if text:
                # 문장 분리 후 앞부분만 요약처럼 사용
                parts = re.split(r"(?<=[.!?])\\s+", text)
                preview = " ".join(parts[:4]).strip()
                if preview:
                    return preview
    except Exception:
        pass

    # 2) 요약 API 시도
    try:
        r = requests.get(WIKI_SUMMARY_URL.format(title=_urlquote(entity)), timeout=10)
        if r.status_code == 200:
            data = r.json()
            extract = data.get("extract")
            if extract:
                return extract
    except Exception:
        pass

    # 3) 폴백: opensearch
    try:
        params = {
            "action": "opensearch",
            "search": entity,
            "limit": 5,
            "namespace": 0,
            "format": "json",
        }
        r = requests.get(WIKI_OPENSEARCH_URL, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and len(data) >= 4:
                titles = data[1]
                descs = data[2]
                if titles and descs:
                    # 첫 번째 비어있지 않은 설명 반환
                    for d in descs:
                        if d:
                            return d
                if titles:
                    return f"No summary found. Similar entities: {', '.join(titles)}"
    except Exception:
        pass

    return "No relevant Wikipedia content found."


def web_search(query: str) -> str:
    """웹 검색을 수행합니다. Bing API가 있으면 사용, 없으면 위키 오픈서치로 폴백."""
    query = (query or "").strip()
    if not query:
        return "No query provided."

    api_key = os.getenv("BING_API_KEY")
    if api_key:
        try:
            headers = {"Ocp-Apim-Subscription-Key": api_key}
            params = {"q": query, "textDecorations": True, "textFormat": "HTML"}
            r = requests.get(
                BING_SEARCH_URL, headers=headers, params=params, timeout=10
            )
            if r.status_code == 200:
                data = r.json()
                # 검색 결과 스니펫 상위 5개 취합
                snippets: List[str] = []
                for item in data.get("webPages", {}).get("value", [])[:5]:
                    name = item.get("name", "")
                    snippet = item.get("snippet", "")
                    url = item.get("url", "")
                    if snippet:
                        snippets.append(f"{name}: {snippet} ({url})")
                if snippets:
                    return "\n".join(snippets)
                return "Search returned no results."
        except Exception:
            # 폴백 경로로 진행
            pass

    # 폴백: 위키 오픈서치(공용 엔드포인트)
    try:
        params = {
            "action": "opensearch",
            "search": query,
            "limit": 5,
            "namespace": 0,
            "format": "json",
        }
        r = requests.get(WIKI_OPENSEARCH_URL, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and len(data) >= 4:
                titles = data[1] or []
                descs = data[2] or []
                links = data[3] or []
                lines = []
                for t, d, l in zip(titles, descs, links):
                    lines.append(f"{_html_unescape(t)}: {_html_unescape(d)} ({l})")
                if lines:
                    return "\n".join(lines)
                return "No search results found."
    except Exception:
        pass

    return "Search failed and no fallback available."


def lookup_keyword(last_text: str, keyword: str) -> str:
    """마지막 패시지에서 키워드를 포함한 문장을 찾아 반환합니다."""
    if not last_text:
        return "No prior passage to lookup from."
    if not keyword:
        return "No keyword provided for lookup."

    # 간단한 문장 분리 (의존성 없이 동작하도록 최소화)
    sentences = re.split(r"(?<=[.!?])\\s+", last_text)
    keyword_lower = keyword.lower()
    for s in sentences:
        if keyword_lower in s.lower():
            return s
    return "No sentence containing the keyword found in the last passage."


def parse_action(string):
    """LLM의 Action 출력 문자열을 파싱하여 Action 타입과 인자를 추출합니다.
    정확히 파싱되지 않으면 fuzzy_parse_action을 시도합니다.
    """
    pattern = r'^(\w+)\[(.+)\]$'

    match = re.match(pattern, string)

    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument
    else:
        action_type, argument = fuzzy_parse_action(string)
        return action_type, argument

def fuzzy_parse_action(text):
    """Action 문자열을 유연하게 파싱하여 Action 타입과 인자를 추출합니다.
    정규 표현식에 엄격하게 일치하지 않아도 시도합니다.
    """
    text = text.strip(' ').strip('.')
    pattern = r'^(\w+)\[(.+)\]'
    match = re.match(pattern, text)
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument
    else:
        return text, ''