"""에이전트가 사용하는 도구(Tool)들의 명세(JSON 스키마)를 정의합니다."""

search_items = {
    "name": "Search",
    "description": "Use this function to search for the target item in the inventory based on keywords",
    "parameters": {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "string",
                "description": "The keywords that describe the item to be searched for"
            },
            "max_price": {
                "type": "string",
                "description": "The upper bound of the item price, if the upper bound is not specified, then set to 1000000.",
            }
        },
        "required": ["keywords"]
    }
}

select_item = {
    "name": "select_item",
    "description": "Use this function to select one of the items from the search results and check its details",
    "parameters": {
        "type": "object",
        "properties": {
            "item_id": {
                "type": "string",
                "description": "The id of the item to be checked"
            },
        },
        "required": ["item_id"]
    }
}

description = {
    "name": "description",
    "description": "Use this function to check the description of the item, if you are unsure if the item perfectly matches the user instruction",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

features = {
    "name": "features",
    "description": "Use this fucntion to check the features of the item, if you are unsure if the item perfectly matches the user instruction",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

reviews = {
    "name": "reviews",
    "description": "Use this function to check the reviews of the item, if you are unsure if the item perfectly matches the user instruction",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

buy_now = {
    "name": "buy_now",
    "description": "Use this function to buy the current item, if the current item perfectly matches the user instruction.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

# 참고: buy_item_final은 buy_now와 이름이 같아 하나로 통일합니다.


previous_page = {
    "name": "previous_page",
    "description": "Use this fucntion to go back to the results page, if the current item does not match the user instruction.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

next_page = {
    "name": "next_page",
    "description": "Use this function to go to the next page of search results to view more items, if none of the items on the current page match the user instruction.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

back_to_search = {
    "name": "back_to_search",
    "description": "Use this function to go back to the initial search page. You should use this function only if you have browsed mutliple pages of items and checked multiple items' details in the history, and none of the items match the user instruction.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}
