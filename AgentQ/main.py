"""
AgentQ CLI 실행 스크립트
"""

import asyncio
import argparse
import sys
import os
from typing import Optional

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentq.graph import get_agentq_executor
from agentq.llm_utils import setup_default_llms, test_llm_connection
from agentq.playwright_helper import connect_to_chrome, cleanup


async def setup_environment():
    """환경 설정 및 초기화"""
    print("🔧 AgentQ 환경 설정 중...")
    
    # 1. LLM 설정
    print("\n1️⃣ LLM 설정...")
    setup_default_llms()
    
    # LLM 연결 테스트
    llm_available = await test_llm_connection()
    if not llm_available:
        print("❌ LLM 연결에 실패했습니다.")
        print("💡 해결 방법:")
        print("   - OpenAI: OPENAI_API_KEY 환경변수 설정")
        print("   - Ollama: ollama serve 명령으로 서버 실행")
        return False
    
    # 2. Chrome 연결
    print("\n2️⃣ Chrome 연결...")
    page = await connect_to_chrome()
    if not page:
        print("❌ Chrome 연결에 실패했습니다.")
        print("💡 해결 방법:")
        print("   - ./scripts/setup_chrome.sh 실행")
        print("   - Chrome이 포트 9222에서 실행 중인지 확인")
        return False
    
    print("✅ Chrome 연결 성공")
    
    print("\n🎯 AgentQ 준비 완료!")
    return True


async def run_agentq(
    user_input: str, 
    max_loops: int = 5, 
    stream: bool = False,
    session_id: Optional[str] = None
):
    """AgentQ 실행"""
    
    # 환경 설정
    if not await setup_environment():
        return False
    
    try:
        # AgentQ 실행기 가져오기
        executor = get_agentq_executor()
        
        if stream:
            # 스트리밍 실행
            await executor.stream_execute(
                user_input=user_input,
                max_loops=max_loops,
                session_id=session_id
            )
        else:
            # 일반 실행
            final_state = await executor.execute(
                user_input=user_input,
                max_loops=max_loops,
                session_id=session_id
            )
            
            # 결과 출력
            print("\n" + "="*60)
            print("📋 최종 결과")
            print("="*60)
            
            if final_state.get("explanation"):
                print(f"💬 답변: {final_state['explanation']}")
            
            if final_state.get("current_url"):
                print(f"🌐 최종 URL: {final_state['current_url']}")
            
            if final_state.get("page_title"):
                print(f"📄 페이지 제목: {final_state['page_title']}")
            
            print(f"🔄 실행된 루프: {final_state['loop_count']}/{final_state['max_loops']}")
            
            if final_state.get("last_error"):
                print(f"⚠️ 마지막 오류: {final_state['last_error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ AgentQ 실행 중 오류: {str(e)}")
        return False
    
    finally:
        # 리소스 정리
        print("\n🧹 리소스 정리 중...")
        await cleanup()


async def interactive_mode():
    """대화형 모드"""
    print("🤖 AgentQ 대화형 모드")
    print("=" * 40)
    print("명령어:")
    print("  - 'quit' 또는 'exit': 종료")
    print("  - 'help': 도움말")
    print("  - 'graph': 그래프 구조 보기")
    print("=" * 40)
    
    # 환경 설정
    if not await setup_environment():
        return
    
    executor = get_agentq_executor()
    session_id = "interactive_session"
    
    try:
        while True:
            print("\n" + "-" * 40)
            user_input = input("🎯 명령을 입력하세요: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 AgentQ를 종료합니다.")
                break
            
            elif user_input.lower() == 'help':
                print("""
AgentQ 사용법:
- 웹 검색: "파이썬이 뭐야?", "서울 날씨 알려줘"
- 페이지 이동: "구글로 이동해줘", "네이버 뉴스 보여줘"
- 정보 수집: "현재 페이지 정보 알려줘"
- 스크린샷: "화면 캡처해줘"
                """)
                continue
            
            elif user_input.lower() == 'graph':
                print(executor.get_graph_visualization())
                continue
            
            # AgentQ 실행
            print(f"\n🚀 실행 중: {user_input}")
            await run_agentq(user_input, session_id=session_id)
    
    except KeyboardInterrupt:
        print("\n\n👋 사용자가 중단했습니다.")
    
    finally:
        await cleanup()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="AgentQ - Advanced AI Web Agent")
    
    parser.add_argument(
        "command", 
        nargs="?", 
        help="실행할 명령 (생략하면 대화형 모드)"
    )
    
    parser.add_argument(
        "--max-loops", 
        type=int, 
        default=5, 
        help="최대 루프 횟수 (기본값: 5)"
    )
    
    parser.add_argument(
        "--stream", 
        action="store_true", 
        help="스트리밍 모드로 실행"
    )
    
    parser.add_argument(
        "--session-id", 
        type=str, 
        help="세션 ID (선택사항)"
    )
    
    args = parser.parse_args()
    
    print("🤖 AgentQ - Advanced AI Web Agent")
    print("=" * 50)
    
    if args.command:
        # 단일 명령 실행
        asyncio.run(run_agentq(
            user_input=args.command,
            max_loops=args.max_loops,
            stream=args.stream,
            session_id=args.session_id
        ))
    else:
        # 대화형 모드
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()