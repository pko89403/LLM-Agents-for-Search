import argparse
import logging
import os
from typing import Optional

from dotenv import load_dotenv

from graph import run_knowagent
from llm_utils import get_default_llm


def main():
    """KnowAgent LangGraph 에이전트 실행용 CLI 엔트리 포인트"""
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # .env 로드 (있으면)
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="KnowAgent LangGraph 에이전트를 실행합니다."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="답변할 질문. 생략 시 --question 또는 $QUESTION 환경변수를 사용합니다.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", "ollama:gemma3:4b"),
        help="사용할 LLM 모델 이름 (기본: 환경변수 OLLAMA_MODEL 또는 ollama:gemma3:4b)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="LLM temperature 값 (기본: 0.0)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=25, help="그래프 최대 스텝(재귀) 수 (기본: 25)"
    )
    parser.add_argument(
        "--question",
        dest="question_flag",
        default=os.getenv("QUESTION"),
        help="질문(위치 인자 대신 옵션으로 제공할 때 사용)",
    )
    parser.add_argument(
        "--max-consec-search",
        type=int,
        default=3,
        help="연속 Search 액션 최대 허용 횟수 (기본: 3)",
    )
    parser.add_argument(
        "--auto-finish-step", type=int, default=6, help="자동 종료 스텝 수 (기본: 6)"
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=2000,
        help="스크래치패드 최대 토큰 길이 (기본: 2000)",
    )
    args = parser.parse_args()

    question: Optional[str] = args.question or args.question_flag
    if not question:
        logging.error(
            "질문이 제공되지 않았습니다. 위치 인자 또는 --question, 혹은 $QUESTION 을 사용하세요."
        )
        raise SystemExit(
            "질문이 제공되지 않았습니다. 위치 인자 또는 --question, 혹은 $QUESTION 을 사용하세요."
        )

    logging.info(f"KnowAgent 시작: 질문='{question}'")
    logging.info(f"모델: {args.model}, Temperature: {args.temperature}")
    logging.info(
        f"최대 스텝: {args.max_steps}, 연속 Search 제한: {args.max_consec_search}, 자동 종료 스텝: {args.auto_finish_step}, 컨텍스트 길이: {args.context_len}"
    )

    # LLM 생성
    logging.info("LLM 생성 중...")
    llm = get_default_llm(model=args.model, temperature=args.temperature)
    logging.info("LLM 생성 완료.")

    logging.info("에이전트 실행 중...")
    result = run_knowagent(
        question=question,
        llm=llm,
        max_steps=args.max_steps,
        max_consec_search=args.max_consec_search,
        auto_finish_step=args.auto_finish_step,
        context_len=args.context_len,
    )
    logging.info("에이전트 실행 완료.")

    print("=== Final Answer ===")
    print(result.get("answer"))

    print("\n=== Scratchpad ===")
    print(result.get("scratchpad"))


if __name__ == "__main__":
    main()
