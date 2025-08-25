# -*- coding: utf-8 -*-
"""에이전트가 사용하는 외부 도구(웹 환경과의 상호작용)를 구현합니다.

ToolKit 클래스는 환경(env)과 상호작용하는 모든 도구의 실행을 관리하는
단일 인터페이스를 제공합니다.
"""

from typing import Any, Dict


class ToolKit:
    """환경과 상호작용하는 도구들의 집합을 관리하고 실행합니다."""

    def __init__(self, env: Any):
        """
        Args:
            env: step(action: str) 메소드를 가진 환경 객체.
        """
        self.env = env
        # LLM이 호출할 수 있는 action 이름과 실제 실행 함수를 매핑합니다.
        self._tool_map = {
            "search": self._search,
            "select_item": self._click_item,
            "description": self._click_description,
            "features": self._click_features,
            "reviews": self._click_reviews,
            "buy_now": self._buy_now,
            "previous_page": self._previous_page,
            "next_page": self._next_page,
            "back_to_search": self._back_to_search,
        }

    def execute(self, action: Dict[str, Any]) -> tuple:
        """LLM의 action 지시를 받아 적절한 도구를 실행합니다."""
        action_name = (action.get("name") or "").lower()
        arguments = action.get("arguments", {})

        tool_func = self._tool_map.get(action_name)

        if not tool_func:
            raise ValueError(f"알 수 없는 도구입니다: {action_name}")

        # LLM이 생성한 arguments 딕셔너리를 그대로 키워드 인자로 전달합니다.
        return tool_func(**arguments)

    # --- 각 도구의 실제 구현 (private 메소드) ---

    def _search(self, keywords: str, **kwargs) -> tuple:
        """환경에 검색 액션을 수행합니다."""
        action_str = f"search[{keywords}]"
        return self.env.step(action_str)

    def _click_item(self, item_id: str, **kwargs) -> tuple:
        """ID를 기반으로 특정 아이템을 클릭합니다."""
        action_str = f"click[{item_id}]"
        return self.env.step(action_str)

    def _click_description(self, **kwargs) -> tuple:
        return self.env.step("click[description]")

    def _click_features(self, **kwargs) -> tuple:
        return self.env.step("click[features]")

    def _click_reviews(self, **kwargs) -> tuple:
        return self.env.step("click[reviews]")

    def _buy_now(self, **kwargs) -> tuple:
        """환경에 구매(Buy Now) 액션을 수행합니다."""
        return self.env.step("click[Buy Now]")

    def _previous_page(self, **kwargs) -> tuple:
        """환경에 이전 페이지로 가는 액션을 수행합니다."""
        return self.env.step("click[< Prev]")

    def _next_page(self, **kwargs) -> tuple:
        """환경에 다음 페이지로 가는 액션을 수행합니다."""
        return self.env.step("click[Next >]")

    def _back_to_search(self, **kwargs) -> tuple:
        """환경에 검색 페이지로 돌아가는 액션을 수행합니다."""
        return self.env.step("click[Back to Search]")
