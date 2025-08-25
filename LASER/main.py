# -*- coding: utf-8 -*-
"""에이전트를 실행하는 메인 스크립트입니다."""

import argparse
import os
import sys
import pprint
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

if 'INSTRUCTION' in os.environ:
    del os.environ['INSTRUCTION']

from llm_utils import get_default_llm
from graph import run_laser_agent
from replay import OfflineWebshopEnv

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="LASER 에이전트 실행 스크립트")

    # 위치 인자 (선택적)
    parser.add_argument(
        "instruction_pos",
        metavar="instruction",
        nargs="?",
        type=str,
        help="에이전트에게 내릴 지시사항 (위치 인자 방식)",
    )

    # 키워드 인자
    parser.add_argument("--instruction", "-i", type=str, default=None, help="에이전트에게 내릴 지시사항 (--instruction 방식)")
    parser.add_argument("--model", type=str, default="llama3.2:3b", help="사용할 LLM 모델 (예: gpt-4o-mini, ollama:mistral)")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM의 temperature 설정")
    parser.add_argument("--max-steps", type=int, default=15, help="에이전트의 최대 스텝 수")
    parser.add_argument("--session-id", type=int, default=3, help="리플레이할 WebShop 데모 세션 ID (리플레이 모드에서만 유효)")
    parser.add_argument("--demo-file", type=str, default="webshop_demonstrations_0-100.json", help="WebShop 데모 파일 경로 (리플레이 모드에서만 유효)")
    parser.add_argument("--enable-feedback", action="store_true", help="피드백 시스템을 활성화합니다 (매니저 피드백 및 재고 기능)")

    # 모드 선택 인자
    parser.add_argument(
        "--mode",
        type=str,
        choices=["replay", "real"],
        default="replay",
        help="실행 모드: 'replay' (기록된 데모), 'real' (실제 웹 환경)",
    )

    args = parser.parse_args()

    # --- 지시사항 및 모드 검증 ---
    if args.mode == "replay":
        # 리플레이 모드에서는 외부 지시사항 입력을 허용하지 않음
        if args.instruction or args.instruction_pos or os.getenv("INSTRUCTION"):
            print("오류: 'replay' 모드에서는 --instruction, 위치 인자, 또는 환경 변수를 통한 지시사항 입력을 허용하지 않습니다.", file=sys.stderr)
            print("지시사항은 데모 파일에서 자동으로 로드됩니다.", file=sys.stderr)
            sys.exit(1)
    elif args.mode == "real":
        # 리얼 모드에서는 지시사항이 필수
        if not (args.instruction or args.instruction_pos or os.getenv("INSTRUCTION")):
            print("오류: 'real' 모드에서는 --instruction, 위치 인자, 또는 환경 변수를 통해 지시사항을 반드시 입력해야 합니다.", file=sys.stderr)
            sys.exit(1)

    # --- 환경 초기화 및 지시사항 설정 ---
    env = None
    instruction_for_agent = None
    initial_observation = None

    if args.mode == "replay":
        print("▶ 모드: 리플레이(Replay)")
        env = OfflineWebshopEnv(args.demo_file)
        instruction_for_agent = env.session_map.get(args.session_id, {}).get('instruction')

        if instruction_for_agent is None:
            print(f"오류: 세션 {args.session_id}의 지시사항을 데모 파일에서 찾을 수 없습니다.", file=sys.stderr)
            sys.exit(1)

        # 리플레이 환경 리셋 및 초기 관찰 획득
        initial_observation = env.reset(session_id=args.session_id)
        if initial_observation is None:
            print(f"오류: 세션 {args.session_id}의 초기 관찰을 얻을 수 없습니다.", file=sys.stderr)
            sys.exit(1)

    elif args.mode == "real":
        print("▶ 모드: 리얼(Real)")
        instruction_for_agent = args.instruction_pos or args.instruction or os.getenv("INSTRUCTION")
        
        # TODO: 실제 웹 환경 연동
        print("  - 실제 웹 환경 연동이 필요합니다. 현재는 미구현입니다.", file=sys.stderr)
        sys.exit(1)

    # 환경 객체가 제대로 생성되지 않았을 경우
    if env is None:
        print("오류: 환경을 초기화할 수 없습니다. 모드 설정을 확인하세요.", file=sys.stderr)
        sys.exit(1)

    print(f"▶ 지시사항: {instruction_for_agent}")
    print(f"▶ 세션 ID: {args.session_id} (리플레이 모드에서만 유효)")

    # 2. LLM 초기화
    try:
        llm = get_default_llm(model=args.model, temperature=args.temperature)
        print(f"▶ LLM: {llm.model if hasattr(llm, 'model') else llm.__class__.__name__}")
    except RuntimeError as e:
        print(f"LLM 초기화 오류: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. 에이전트 실행
    final_state = run_laser_agent(
        env=env,
        instruction=instruction_for_agent,
        initial_observation=initial_observation, # 추가
        initial_url=None, # 추가 (리플레이 모드에서는 URL이 필요 없음)
        llm=llm,
        max_steps=args.max_steps,
        session_id=args.session_id, # 추가: 세션 ID 전달
        enable_feedback=args.enable_feedback # 추가: 피드백 시스템 활성화 여부
    )

    # 4. 최종 결과 출력
    print("\n" + "="*50)
    print("[최종 실행 결과]")
    pprint.pprint(final_state)
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
