# -*- coding: utf-8 -*-
"""WebShop 환경의 관찰(observation) 문자열을 파싱하는 유틸리티 함수들을 정의합니다."""

import re
from typing import Any, Dict, List


def _parse_item_block(lines: List[str], parsed_data: Dict):
    """아이템 블록을 파싱합니다.
    관찰 형식 예시 (Result 페이지):
        [button] B09MW563KN [button_]
        SWAGOFKGys Travel Toothbrushes ...
        $22.9
    관찰 형식 예시 (Item 페이지):
        SWAGOFKGys Travel Toothbrushes ...
        Price: $22.9
    위 두 경우 모두를 처리합니다.
    """
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()

        # 1) 결과 목록에서의 아이템: "[button] <ID> [button_]" 다음 줄이 이름, 그 다음 줄이 가격
        m = re.match(r"\[button\]\s*(B\w+)\s*\[button_]", line)
        if m:
            item_id = m.group(1).strip()
            # 이름은 같은 라인에 올 수도, 다음 라인에 올 수도 있다.
            # 같은 라인에 있는 경우: "[button] <ID> [button_] <NAME>"
            after_button = line[m.end():].strip()
            name = ""
            if after_button and not after_button.lower().startswith("price:") and not after_button.startswith("$"):
                name = after_button
            elif i + 1 < n:
                name = lines[i + 1].strip()
            price_str = ""
            # 가격 줄은 두 가지 패턴을 허용: "$12.34" 또는 "Price: $12.34 ..."
            j = i + 2
            while j < n:
                cand = lines[j].strip()
                if cand.startswith("$"):
                    price_str = cand[1:].strip()
                    break
                if cand.lower().startswith("price:"):
                    # 예: "Price: $22.9" 또는 "Price: $12.00 to $15.00"
                    m_price = re.search(r"\$([\d\.,]+(?:\s*to\s*\$[\d\.,]+)?)", cand, re.IGNORECASE)
                    if m_price:
                        price_str = m_price.group(1)
                        break
                if cand.startswith("[button]"):
                    # 다음 아이템 혹은 네비게이션 버튼을 만나면 중단
                    break
                j += 1

            parsed_data["items"].append({
                "item_id": item_id,
                "name": name,
                "price_str": price_str,
            })
            # 다음 탐색은 j로 이동(가격 줄까지 소모)하되, 최소 1은 전진
            i = max(j, i + 1)
            continue

        # 2) 아이템 상세 페이지에서의 아이템: 버튼 없이 제목/가격만 있는 경우 처리 (보조적)
        #    이 경우에는 item_id를 알 수 없으므로 생략하거나 ID 없이 기록
        if line.lower().startswith("price:") or line.startswith("$"):
            # 바로 앞 줄이 이름일 가능성이 큼
            name = lines[i - 1].strip() if i - 1 >= 0 else ""
            price_str = line[1:].strip() if line.startswith("$") else (
                re.search(r"\$([\d\.,]+(?:\s*to\s*\$[\d\.,]+)?)", line, re.IGNORECASE).group(1)
                if re.search(r"\$([\d\.,]+(?:\s*to\s*\$[\d\.,]+)?)", line, re.IGNORECASE) else ""
            )
            parsed_data["items"].append({
                "item_id": None,
                "name": name,
                "price_str": price_str,
            })
        i += 1


def _parse_customization_block(lines: List[str], parsed_data: Dict):
    """커스터마이징 블록을 파싱합니다."""
    block_text = "\n".join(lines)
    customization_pattern = re.compile(r"^(\\w+):\n((?:\s*\\[button\\]\s*.*?\s*\\[button_\\\\]\n)+", re.MULTILINE)
    for match in customization_pattern.finditer(block_text):
        custom_type = match.group(1).strip()
        choices_block = match.group(2)
        choices = re.findall(r"\\[button\\]\s*(.*?)\s*\\[button_]", choices_block)
        parsed_data["customizations"][custom_type] = [c.strip() for c in choices]

def _parse_target_instruction(instruction: str) -> Dict[str, Any]:
    """사용자 지시사항에서 목표 아이템의 키워드와 최대 가격을 파싱합니다."""
    keywords = []
    max_price = None

    # 가격 파싱 (예: "price lower than 50.00 dollars")
    price_match = re.search(r"price (?:lower than|under) ([\d\.]+)(?: dollars)?", instruction, re.IGNORECASE)
    if price_match:
        max_price = float(price_match.group(1))
        # 가격 정보는 키워드에서 제외
        instruction = re.sub(r"price (?:lower than|under) [\d\.]+(?: dollars)?", "", instruction, flags=re.IGNORECASE)

    # 기타 키워드 추출 (간단하게 단어 단위로)
    # TODO: 더 정교한 키워드 추출 필요 (예: spaCy 사용)
    # 불필요한 구두점 제거 후 공백으로 분리
    keywords = [word.strip() for word in re.split(r"\s+|[.,;!?]", instruction) if word.strip()]

    return {"keywords": keywords, "max_price": max_price}

def parse_observation(obs: str) -> Dict[str, Any]:
    """WebShop 환경의 관찰(observation) 문자열을 파싱합니다."""
    parsed_data = {
        "buttons": [],
        "items": [],
        "page_info": {},
        "customizations": {},
        "description_viewed": False,
        "features_viewed": False,
        "reviews_viewed": False,
        "raw_obs": obs, # 원본 관찰 문자열도 저장
        "item_details_text": "" # Item 상태에서 아이템 상세 정보 텍스트
    }

    if obs is None or not obs.strip(): # None 또는 빈 문자열 처리
        return parsed_data

    lines = obs.splitlines()
    current_section_type = None
    current_section_lines = []

    def process_section(section_type, section_lines, p_data):
        if not section_lines: return

        block_content = "\n".join(section_lines)

        if section_type == "buttons":
            # 버튼 영역: "[button] ... [button_]" 또는 "[clicked button] ... [clicked button_]" 모두 지원
            button_matches = re.findall(r"\[(clicked )?button\]\s*(.*?)\s*\[(?:clicked )?button_]", block_content)
            for clicked_prefix, text in button_matches:
                p_data["buttons"].append({
                    "text": text.strip(),
                    "clicked": bool(clicked_prefix)
                })
            # 결과 페이지의 아이템은 버튼 라인과 같은 블록에 존재하므로 여기서도 아이템 파싱을 수행한다.
            _parse_item_block(section_lines, p_data)
        elif section_type == "page_info":
            # 예: "Page 1 (Total results: 50)" 또는 "Page 1"
            page_info_match = re.search(r"Page (\d+)(?: \(Total results: (\d+)\))?", block_content)
            if page_info_match:
                total = page_info_match.group(2)
                p_data["page_info"] = {
                    "current_page": int(page_info_match.group(1)),
                    "total_results": int(total) if total is not None else None,
                }
        elif section_type == "item_block":
            _parse_item_block(section_lines, p_data)
        elif section_type == "customization_block":
            _parse_customization_block(section_lines, p_data)
        elif section_type == "item_details":
            p_data["item_details_text"] = block_content
            if "description:" in block_content and "description: (if this is shown" not in block_content:
                p_data["description_viewed"] = True
            if "features:" in block_content and "features: (if this is shown" not in block_content:
                p_data["features_viewed"] = True
            if "reviews:" in block_content and "reviews: (if this is shown" not in block_content:
                p_data["reviews_viewed"] = True

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            if current_section_type:
                process_section(current_section_type, current_section_lines, parsed_data)
                current_section_lines = []
                current_section_type = None
            continue

        if stripped_line.startswith("[button]") or stripped_line.startswith("[clicked button]"):
            if current_section_type != "buttons":
                process_section(current_section_type, current_section_lines, parsed_data)
                current_section_lines = []
            current_section_type = "buttons"
        elif re.match(r"Page \d+( \(Total results: \d+\))?", stripped_line):
            if current_section_type != "page_info":
                process_section(current_section_type, current_section_lines, parsed_data)
                current_section_lines = []
            current_section_type = "page_info"
        elif re.match(r"^\\w+:\s*$", stripped_line) and not stripped_line.startswith("Instruction:"):
            if current_section_type != "customization_block":
                process_section(current_section_type, current_section_lines, parsed_data)
                current_section_lines = []
            current_section_type = "customization_block"
        elif "description:" in stripped_line or "features:" in stripped_line or "reviews:" in stripped_line:
            if current_section_type != "item_details":
                process_section(current_section_type, current_section_lines, parsed_data)
                current_section_lines = []
            current_section_type = "item_details"
        elif "Instruction:" in stripped_line:
            if current_section_type != "instruction":
                process_section(current_section_type, current_section_lines, parsed_data)
                current_section_lines = []
            current_section_type = "instruction"
        elif current_section_type is None:
            if not any(keyword in stripped_line for keyword in ["Instruction:", "Page ", "description:", "features:", "reviews:"]) and not stripped_line.startswith("["):
                current_section_type = "item_block"
            else:
                current_section_type = "other"

        current_section_lines.append(line)

    if current_section_type:
        process_section(current_section_type, current_section_lines, parsed_data)

    return parsed_data
