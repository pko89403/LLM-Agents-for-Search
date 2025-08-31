from agentq.graph import AgentQExecutor
import asyncio

# main.py에서 환경 설정 로직을 가져와 일부 사용합니다.
from agentq.llm_utils import setup_default_llms, test_llm_connection
from agentq.playwright_helper import connect_to_chrome, cleanup

async def main():
    print("--- 환경 설정 시작 ---")
    setup_default_llms()
    if not await test_llm_connection():
        print("LLM 연결 실패. OPENAI_API_KEY 또는 Ollama 설정을 확인하세요.")
        return
    if not await connect_to_chrome():
        print("Chrome 연결 실패. ./scripts/setup_chrome.sh를 실행했는지 확인하세요.")
        return
    print("--- 환경 설정 완료 ---")

    try:
        print('--- AgentQ Executor 초기화 ---')
        ex = AgentQExecutor()
        print('--- 그래프 컴파일 ---')
        ex.compile()
        print('--- OpenTable 테스트 실행 ---')
        final_state = await ex.execute("OpenTable에서 'Cote Korean Steakhouse' 2명 내일 저녁 7시 예약 가능 시간 확인")
        print('--- 최종 상태 ---')
        print(final_state)
    finally:
        await cleanup()

if __name__ == "__main__":
    asyncio.run(main())
