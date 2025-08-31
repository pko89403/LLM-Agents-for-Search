#!/usr/bin/env python3
"""
AgentQ 테스트 실행 스크립트
"""

import asyncio
import argparse
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test.agentq_test_runner import AgentQTestRunner


async def main():
    """메인 실행 함수"""
    
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(
        description="AgentQ 테스트 실행기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python run_agentq_tests.py                           # 기본 테스트 실행
  python run_agentq_tests.py --min 0 --max 3          # 처음 3개 태스크만 실행
  python run_agentq_tests.py --headless False         # 브라우저 UI 표시
  python run_agentq_tests.py --file test/tasks/two_tasks.json  # 특정 파일 사용
        """
    )
    
    parser.add_argument(
        "--file", "-f",
        type=str,
        default="test/tasks/test.json",
        help="테스트 태스크 파일 경로 (기본값: test/tasks/test.json)"
    )
    
    parser.add_argument(
        "--min", "-min",
        type=int,
        default=0,
        help="시작 태스크 인덱스 (기본값: 0)"
    )
    
    parser.add_argument(
        "--max", "-max",
        type=int,
        default=None,
        help="종료 태스크 인덱스 (기본값: 모든 태스크)"
    )
    
    parser.add_argument(
        "--headless",
        type=lambda x: x.lower() == 'true',
        default=True,
        help="헤드리스 모드 (기본값: True)"
    )
    
    parser.add_argument(
        "--wait",
        type=int,
        default=2,
        help="태스크 간 대기 시간(초) (기본값: 2)"
    )
    
    parser.add_argument(
        "--results-id",
        type=str,
        default="",
        help="테스트 결과 ID (기본값: 자동 생성)"
    )
    
    args = parser.parse_args()
    
    # 테스트 파일 존재 확인
    if not os.path.exists(args.file):
        print(f"❌ 테스트 파일을 찾을 수 없습니다: {args.file}")
        print("사용 가능한 테스트 파일:")
        test_dir = "test/tasks"
        if os.path.exists(test_dir):
            for file in os.listdir(test_dir):
                if file.endswith('.json'):
                    print(f"  - {os.path.join(test_dir, file)}")
        return 1
    
    print("🚀 AgentQ 테스트 시작")
    print("=" * 60)
    print(f"📂 테스트 파일: {args.file}")
    print(f"📊 태스크 범위: {args.min} ~ {args.max or '끝까지'}")
    print(f"🖥️ 헤드리스 모드: {args.headless}")
    print(f"⏱️ 대기 시간: {args.wait}초")
    print("=" * 60)
    
    try:
        # 테스트 실행기 생성 및 실행
        runner = AgentQTestRunner()
        
        results = await runner.run_tests(
            test_file=args.file,
            min_task_index=args.min,
            max_task_index=args.max,
            test_results_id=args.results_id,
            headless=args.headless,
            wait_time=args.wait
        )
        
        # 최종 결과 출력
        print(f"\n🎉 테스트 완료! 총 {len(results)}개 태스크 실행됨")
        
        # 성공률 계산
        passed = len([r for r in results if r["score"] == 1])
        success_rate = (passed / len(results)) * 100 if results else 0
        
        print(f"✅ 성공률: {success_rate:.1f}% ({passed}/{len(results)})")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단됨")
        return 1
        
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)