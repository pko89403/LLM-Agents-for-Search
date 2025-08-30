
'''
ITTS 에이전트 실행을 위한 CLI 스크립트
'''
import argparse
from dotenv import load_dotenv
from graph import run_agent
from llm_utils import get_llm_manager

def main():
    """메인 실행 함수"""
    load_dotenv()

    parser = argparse.ArgumentParser(description="ITTS WebShop 에이전트")
    parser.add_argument("--goal", type=str, help="에이전트의 목표 (예: 'Find a camera under $100')")
    parser.add_argument("--max-steps", type=int, default=3, help="최대 탐색 깊이 (d)")
    parser.add_argument("--branching", type=int, default=2, help="브랜치 요소 (b): 액션 및 가치 함수 샘플링 수")
    parser.add_argument("--budget", type=int, default=5, help="탐색 예산 (c): 최대 노드 확장 수")

    args = parser.parse_args()

    print("🧠 ITTS WebShop 에이전트 시작")

    llm_manager = get_llm_manager()
    client_info = llm_manager.get_client_info()
    print(f"🤖 LLM: {client_info['provider']} - {client_info['model']}")
    print(f"🌳 브랜칭 팩터 (b): {args.branching}")
    print(f"📏 최대 깊이 (d): {args.max_steps}")
    print(f"💰 탐색 예산 (c): {args.budget}")

    if args.goal:
        run_single_goal(args.goal, args.max_steps, args.branching, args.budget)
    else:
        run_demo_mode(args.max_steps, args.branching, args.budget)

def run_single_goal(goal: str, max_steps: int, branching: int, budget: int):
    """단일 목표를 실행합니다."""
    print(f"\n🎯 목표: {goal}")
    try:
        result = run_agent(goal, max_steps, branching, budget)
        print_result(result)
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        if 'result' in locals() and result:
            print_result(result)

def run_demo_mode(max_steps: int, branching: int, budget: int):
    """WebShop 시나리오 데모를 실행합니다."""
    print("\n🎪 데모 모드 실행")



    demo_goals = [
        "Find a durable camera under $100",
        "I need a pair of men's walking shoes, size 10, brand 'Nike'",
        "Find the cheapest laptop with at least 16GB of RAM"
    ]

    for i, goal in enumerate(demo_goals, 1):
        print(f"\n--- 데모 {i}: {goal} ---")
        print(f"목표: '{goal}', 최대 스텝: {max_steps}, 브랜칭 팩터: {branching}, 탐색 예산: {budget}")
        try:
            result = run_agent(goal, max_steps, branching, budget)
            print_result(result)
        except Exception as e:
            print(f"❌ 실행 중 오류 발생: {e}")

    print("\n🎉 모든 데모 완료!")

def print_result(result: dict):
    """ITTS 에이전트의 최종 결과를 출력합니다."""
    print("\n✅ 실행 완료!")

    best_state = result.get("best_state")
    if best_state:
        print(f"   최고 점수: {result.get('best_score', 0.0):.4f}")
        print(f"   탐색 스텝: {result.get('search_counter', 0)}")
        print("\n--- 최적 경로 ---")
        for i, action in enumerate(best_state.get('action_history', [])):
            print(f"   {i+1}. {action}")

        final_answer = result.get('final_answer')
        if final_answer:
            print(f"\n💡 최종 답변: {final_answer}")
        else:
            print("\n💡 최종 답변: (없음)")
    else:
        print("\n⚠️ 유의미한 결과를 찾지 못했습니다.")

if __name__ == "__main__":
    main()
