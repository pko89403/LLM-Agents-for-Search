"""
AgentQ와 테스트 시스템 통합 실행기
"""

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from playwright.async_api import async_playwright, Page, Browser
from tabulate import tabulate
from termcolor import colored

from agentq.graph import get_agentq_executor
from agentq.state import AgentState
from agentq.playwright_helper import PlaywrightHelper
from test.evaluators import evaluator_router
from test.test_utils import (
    get_formatted_current_timestamp,
    load_config,
    task_config_validator,
)

# 테스트 관련 경로
TEST_TASKS = "test/tasks"
TEST_LOGS = "test/logs"
TEST_RESULTS = "test/results"


class AgentQTestRunner:
    """AgentQ 전용 테스트 실행기"""
    
    def __init__(self):
        self.executor = get_agentq_executor()
        self.playwright_helper = None
        self.browser = None
        self.page = None
    
    async def setup_browser(self, headless: bool = True):
        """브라우저 초기화"""
        self.playwright_helper = PlaywrightHelper()
        await self.playwright_helper.setup(headless=headless)
        self.browser = self.playwright_helper.browser
        self.page = self.playwright_helper.page
        print("✅ 브라우저 초기화 완료")
    
    async def cleanup_browser(self):
        """브라우저 정리"""
        if self.playwright_helper:
            await self.playwright_helper.cleanup()
            print("✅ 브라우저 정리 완료")
    
    def create_test_folders(self):
        """테스트 폴더 생성"""
        for folder in [TEST_LOGS, TEST_RESULTS]:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"📁 폴더 생성: {folder}")
    
    def create_task_log_folders(self, task_id: str, test_results_id: str) -> Dict[str, str]:
        """태스크별 로그 폴더 생성"""
        task_log_dir = os.path.join(
            TEST_LOGS, f"{test_results_id}", f"logs_for_task_{task_id}"
        )
        task_screenshots_dir = os.path.join(task_log_dir, "snapshots")
        
        for directory in [task_log_dir, task_screenshots_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        return {
            "task_log_folder": task_log_dir,
            "task_screenshots_folder": task_screenshots_dir,
        }
    
    def save_task_log(self, task_id: str, state: AgentState, logs_dir: str):
        """태스크 실행 로그 저장"""
        log_data = {
            "task_id": task_id,
            "user_input": state.get("user_input"),
            "objective": state.get("objective"),
            "plan": state.get("plan"),
            "final_explanation": state.get("explanation"),
            "loop_count": state.get("loop_count"),
            "done": state.get("done"),
            "scratchpad": state.get("scratchpad", []),
            "last_error": state.get("last_error"),
            "current_url": state.get("current_url"),
            "page_title": state.get("page_title")
        }
        
        file_name = os.path.join(logs_dir, f"agentq_execution_{task_id}.json")
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4)
    
    async def execute_single_task(
        self, 
        task_config: Dict[str, Any],
        logs_dir: str
    ) -> Dict[str, Any]:
        """단일 태스크 실행"""
        
        # 태스크 설정 검증
        task_config_validator(task_config)
        
        task_id = task_config.get("task_id")
        intent = task_config.get("intent", "")
        start_url = task_config.get("start_url")
        
        print(f"\n🎯 태스크 {task_id} 실행 시작")
        print(f"📝 의도: {intent}")
        print(f"🌐 시작 URL: {start_url}")
        
        # 시작 URL로 이동
        if start_url:
            await self.page.goto(start_url, wait_until="load", timeout=30000)
            print(f"✅ {start_url}로 이동 완료")
        
        # AgentQ 실행
        start_time = time.time()
        
        try:
            # PlaywrightHelper를 AgentQ에 전달하기 위해 전역 설정
            import agentq.tools as tools
            tools.set_playwright_helper(self.playwright_helper)
            
            final_state = await self.executor.execute(
                user_input=intent,
                max_loops=5,
                session_id=f"task_{task_id}"
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            print(f"⏱️ 실행 시간: {execution_time:.2f}초")
            print(f"🔄 루프 횟수: {final_state.get('loop_count', 0)}")
            print(f"✅ 완료 상태: {final_state.get('done', False)}")
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"❌ 실행 중 오류: {str(e)}")
            
            # 오류 상태 생성
            final_state = {
                "user_input": intent,
                "done": False,
                "explanation": f"실행 중 오류 발생: {str(e)}",
                "last_error": str(e),
                "loop_count": 0,
                "current_url": self.page.url if self.page else None
            }
        
        # 로그 저장
        self.save_task_log(str(task_id), final_state, logs_dir)
        
        # 태스크 결과 생성
        task_result = {
            "task_id": task_id,
            "intent": intent,
            "start_url": start_url,
            "last_url": self.page.url if self.page else None,
            "tct": execution_time,  # Task Completion Time
            "start_ts": get_formatted_current_timestamp(),
            "completion_ts": get_formatted_current_timestamp(),
            "agentq_explanation": final_state.get("explanation", ""),
            "loop_count": final_state.get("loop_count", 0),
            "done": final_state.get("done", False),
            "error": final_state.get("last_error")
        }
        
        # 평가 실행
        try:
            evaluator = evaluator_router(task_config)
            evaluator_result = await evaluator(
                task_config=task_config,
                page=self.page,
                client=None,  # CDP 세션 없음
                answer=final_state.get("explanation", "")
            )
            
            task_result["score"] = evaluator_result["score"]
            task_result["reason"] = evaluator_result["reason"]
            
        except Exception as e:
            print(f"⚠️ 평가 중 오류: {str(e)}")
            task_result["score"] = -1  # 평가 실패
            task_result["reason"] = f"평가 오류: {str(e)}"
        
        return task_result
    
    def print_task_result(self, task_result: Dict[str, Any], index: int, total: int):
        """태스크 결과 출력"""
        score = task_result["score"]
        if score == 1:
            status, color = "Pass", "green"
        elif score < 0:
            status, color = "Skip", "yellow"
        else:
            status, color = "Fail", "red"
        
        result_table = [
            ["Index", "Task ID", "Intent", "Status", "Time (s)", "Loops"],
            [
                index,
                task_result["task_id"],
                task_result["intent"][:50] + "..." if len(task_result["intent"]) > 50 else task_result["intent"],
                colored(status, color),
                round(task_result["tct"], 2),
                task_result["loop_count"]
            ],
        ]
        print("\n" + tabulate(result_table, headers="firstrow", tablefmt="grid"))
        
        if task_result.get("reason"):
            print(f"📋 평가 이유: {task_result['reason']}")
    
    async def run_tests(
        self,
        test_file: str = "test/tasks/test.json",
        min_task_index: int = 0,
        max_task_index: Optional[int] = None,
        test_results_id: str = "",
        headless: bool = True,
        wait_time: int = 2
    ) -> List[Dict[str, Any]]:
        """테스트 실행"""
        
        print("🚀 AgentQ 테스트 실행 시작")
        print("=" * 60)
        
        # 초기화
        self.create_test_folders()
        await self.setup_browser(headless=headless)
        
        # 테스트 설정 로드
        print(f"📂 테스트 파일 로드: {test_file}")
        test_configurations = load_config(test_file)
        
        if not test_results_id:
            test_results_id = f"agentq_test_{int(time.time())}"
        
        # 테스트 범위 설정
        max_task_index = max_task_index or len(test_configurations)
        total_tests = max_task_index - min_task_index
        
        print(f"📊 총 {total_tests}개 태스크 실행 예정 (인덱스 {min_task_index}~{max_task_index-1})")
        
        test_results = []
        
        try:
            for index, task_config in enumerate(
                test_configurations[min_task_index:max_task_index], 
                start=min_task_index
            ):
                task_id = str(task_config.get("task_id"))
                
                # 로그 폴더 생성
                log_folders = self.create_task_log_folders(task_id, test_results_id)
                
                print(f"\n{'='*60}")
                print(f"📋 태스크 {index + 1}/{total_tests} (ID: {task_id})")
                
                # 태스크 실행
                task_result = await self.execute_single_task(
                    task_config, 
                    log_folders["task_log_folder"]
                )
                
                test_results.append(task_result)
                self.print_task_result(task_result, index + 1, total_tests)
                
                # 대기 시간
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
        
        finally:
            # 정리
            await self.cleanup_browser()
        
        # 결과 요약
        self.print_summary(test_results, total_tests)
        
        # 결과 저장
        self.save_test_results(test_results, test_results_id)
        
        return test_results
    
    def print_summary(self, test_results: List[Dict[str, Any]], total_tests: int):
        """결과 요약 출력"""
        passed_tests = [r for r in test_results if r["score"] == 1]
        failed_tests = [r for r in test_results if 0 <= r["score"] < 1]
        skipped_tests = [r for r in test_results if r["score"] < 0]
        
        print(f"\n{'='*60}")
        print("📊 테스트 결과 요약")
        print("=" * 60)
        
        summary_table = [
            ["항목", "개수", "비율"],
            ["총 테스트", total_tests, "100%"],
            ["성공", len(passed_tests), f"{len(passed_tests)/total_tests*100:.1f}%"],
            ["실패", len(failed_tests), f"{len(failed_tests)/total_tests*100:.1f}%"],
            ["건너뜀", len(skipped_tests), f"{len(skipped_tests)/total_tests*100:.1f}%"],
        ]
        
        print(tabulate(summary_table, headers="firstrow", tablefmt="grid"))
        
        if test_results:
            avg_time = sum(r["tct"] for r in test_results) / len(test_results)
            total_time = sum(r["tct"] for r in test_results)
            avg_loops = sum(r["loop_count"] for r in test_results) / len(test_results)
            
            print(f"\n⏱️ 평균 실행 시간: {avg_time:.2f}초")
            print(f"⏱️ 총 실행 시간: {total_time:.2f}초")
            print(f"🔄 평균 루프 횟수: {avg_loops:.1f}회")
    
    def save_test_results(self, test_results: List[Dict[str, Any]], test_results_id: str):
        """테스트 결과 저장"""
        file_name = os.path.join(TEST_RESULTS, f"agentq_test_results_{test_results_id}.json")
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(test_results, f, ensure_ascii=False, indent=4)
        print(f"💾 테스트 결과 저장: {file_name}")


# 편의 함수
async def run_agentq_tests(
    test_file: str = "test/tasks/test.json",
    min_task_index: int = 0,
    max_task_index: Optional[int] = None,
    headless: bool = True
):
    """AgentQ 테스트 실행 편의 함수"""
    runner = AgentQTestRunner()
    return await runner.run_tests(
        test_file=test_file,
        min_task_index=min_task_index,
        max_task_index=max_task_index,
        headless=headless
    )


if __name__ == "__main__":
    # 예시 실행
    asyncio.run(run_agentq_tests(
        test_file="test/tasks/test.json",
        min_task_index=0,
        max_task_index=3,  # 처음 3개 태스크만 실행
        headless=False  # 브라우저 UI 표시
    ))