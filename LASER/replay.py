# -*- coding: utf-8 -*-
"""Offline Replay Harness

`offline_replay_harness.md` 문서의 명세에 따라, 미리 기록된 데모 로그를 사용하여
에이전트의 행동을 재현하고 테스트하는 시스템을 구현합니다.
"""

import json
from typing import Any, Dict, List, Optional


class OfflineWebshopEnv:
    """미리 생성된 데모 로그 파일을 기반으로 WebShop 환경을 시뮬레이션합니다.

    실제 환경에 접속하지 않고도 에이전트의 행동을 결정론적으로 테스트할 수 있습니다.
    """

    def __init__(self, demo_file_path: str):
        """데모 파일을 로드하고 세션 ID로 에피소드를 인덱싱합니다."""
        print(f"리플레이 환경 초기화: {demo_file_path}")
        try:
            with open(demo_file_path, 'r', encoding='utf-8') as f:
                self.episodes: List[Dict[str, Any]] = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"오류: 데모 파일을 로드할 수 없습니다. 경로를 확인하세요. {e}")
            self.episodes = []

        self.session_map: Dict[int, Dict[str, Any]] = {
            ep['session_id']: ep for ep in self.episodes
        }
        self.current_episode = None
        self.current_step_index = 0
        self.trajectory = []
        self.selected_item_id: Optional[str] = None

    def reset(self, session_id: int) -> Optional[str]:
        """주어진 세션 ID로 환경을 리셋하고 첫 번째 관찰을 반환합니다."""
        if session_id not in self.session_map:
            print(f"오류: 세션 ID {session_id}를 찾을 수 없습니다.")
            return None

        self.current_episode = self.session_map[session_id]
        # 전체 원본 로그를 표준 스텝으로 정규화합니다 (item_page_action 포함)
        raw_traj = self.current_episode.get('trajectory', [])
        self.trajectory = _normalize_trajectory(raw_traj)
        self.current_step_index = 0
        self.selected_item_id = None

        if not self.trajectory:
            print(f"경고: 세션 ID {session_id}에 유효한 trajectory step이 없습니다.")
            return None

        print(f"세션 리셋: {session_id} (총 {len(self.trajectory)} 스텝)")
        # 첫 번째 스텝의 LLM 호출 전 관찰을 반환합니다.
        return self.trajectory[0].get('observation_before_llm')

    def step(self, action_str: str) -> tuple[Optional[str], float, bool, Dict[str, Any]]:
        """에이전트의 액션을 받아 다음 상태로 진행하고 결과를 반환합니다."""
        if not self.current_episode or self.current_step_index >= len(self.trajectory):
            return None, 0.0, True, {'error': '에피소드가 종료되었거나 리셋되지 않았습니다.'}

        current_step_data = self.trajectory[self.current_step_index]

        # 선택 아이템 추적 (select_item 시)
        act_name = (current_step_data.get('llm_action_name') or '').lower()
        if act_name == 'select_item':
            args = current_step_data.get('llm_action_arguments') or {}
            cand = args.get('item_id') or args.get('itemId') or args.get('id')
            if cand:
                self.selected_item_id = str(cand)

        # 현재 스텝의 액션이 LLM이 결정한 액션과 일치하는지 확인
        expected_action = current_step_data.get('action_executed_in_env')
        is_match = action_str.strip() == expected_action.strip() if expected_action else False

        reward = current_step_data.get('reward', 0.0)
        done = current_step_data.get('done', False)

        info = {
            'match': is_match,
            'expected_action': expected_action,
            'predicted_action': action_str,
            'step_number': current_step_data.get('step_number'),
            'index': self.current_step_index,
            'selected_item_id': self.selected_item_id,
        }

        self.current_step_index += 1

        # 다음 스텝의 관찰을 반환합니다.
        next_observation = None
        if self.current_step_index < len(self.trajectory):
            next_observation = self.trajectory[self.current_step_index].get('observation_before_llm', "")
        else:
            # 마지막 스텝이면 최종 관찰을 사용합니다.
            next_observation = current_step_data.get('observation_after_action', "")
            done = True # 마지막 스텝이므로 종료 처리

        return next_observation, reward, done, info

    def get_current_step_info(self) -> Optional[Dict[str, Any]]:
        """리플레이 Runner가 현재 스텝의 정보를 가져오기 위한 헬퍼 함수"""
        if self.current_episode and self.current_step_index < len(self.trajectory):
            return self.trajectory[self.current_step_index]
        return None


def _format_action(step_info: Dict[str, Any]) -> str:
    """로그의 LLM 액션 정보를 환경이 이해하는 action 문자열로 변환합니다."""
    action_name = (step_info.get('llm_action_name') or '').strip()
    args = step_info.get('llm_action_arguments') or {}
    name_ci = action_name.lower()

    if name_ci == 'search':
        return f"search[{args.get('keywords', '')}]"
    elif name_ci in ('select_item','selectitem','click_item','choose_item'):
        return f"click[{args.get('item_id', '')}]"
    elif name_ci == 'description':
        return 'click[description]'
    elif name_ci == 'features':
        return 'click[features]'
    elif name_ci == 'reviews':
        return 'click[reviews]'
    elif name_ci in ('buy_now','buy-now','buynow','buy'):
        return 'click[Buy Now]'
    elif name_ci in ('prev','previous'):
        return 'click[< Prev]'
    elif name_ci in ('next','next_page'):
        return 'click[Next >]'
    elif name_ci in ('back_to_search','back'):
        return 'click[Back to Search]'
    else:
        return ''

def _normalize_trajectory(raw_traj: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """데모 로그의 원본 trajectory를 리플레이 가능한 표준 스텝 목록으로 정규화합니다.
    - step_number가 있는 스텝은 그대로 사용
    - item_page_action(상세 페이지 내 클릭/옵션 선택 등)도 스텝으로 변환하여 포함
    """
    normalized: List[Dict[str, Any]] = []
    for st in raw_traj:
        # 1) 일반 스텝: step_number가 있는 항목
        if 'step_number' in st:
            if not st.get('action_executed_in_env'):
                st['action_executed_in_env'] = _format_action(st)
            normalized.append(st)
            continue

        # 2) 상세 페이지 액션: step_number 없이 type == 'item_page_action'
        if st.get('type') == 'item_page_action':
            llm = st.get('llm_action') or {}
            name = (llm.get('name') or '')
            args = llm.get('arguments', {})
            norm_step: Dict[str, Any] = {
                'step_number': None,  # 기록에 없으므로 None
                'observation_before_llm': st.get('observation_before_action', ''),
                'llm_action_name': name,
                'llm_action_arguments': args,
                'state': 'ItemPage',
                'available_actions': st.get('available_options', []),
                'action_executed_in_env': _format_action({'llm_action_name': name, 'llm_action_arguments': args}),
                'reward': st.get('reward', 0.0),
                'done': st.get('done', False),
                'observation_after_action': st.get('observation_after_action', ''),
            }
            normalized.append(norm_step)
            continue

        # 그 외 포맷은 무시
    return normalized


class StubPolicy:
    """LLM을 호출하는 대신, 로그에 기록된 행동을 그대로 따라하는 정책입니다."""

    def predict_action(self, observation: str, step_info: Dict[str, Any]) -> str:
        """현재 스텝 정보에서 정답 행동을 포매팅하여 반환합니다."""
        if not step_info:
            return "" # 스텝 정보가 없으면 빈 액션 반환
        return _format_action(step_info)


class ReplayRunner:
    """오프라인 리플레이를 실행하고 결과를 리포트합니다."""

    def __init__(self, env: OfflineWebshopEnv, policy: Any):
        self.env = env
        self.policy = policy

    def run_episode(self, session_id: int):
        """하나의 에피소드를 리플레이하고 결과를 출력합니다."""
        obs = self.env.reset(session_id)
        if obs is None:
            return

        done = False
        total_steps = 0
        matched_steps = 0

        while not done:
            step_info = self.env.get_current_step_info()
            if not step_info:
                break

            action_pred = self.policy.predict_action(obs, step_info)
            obs, reward, done, info = self.env.step(action_pred)

            total_steps += 1
            if info.get('match'):
                matched_steps += 1
                label = info.get('step_number')
                if label is None:
                    label = info.get('index')
                print(f"  [Step {label}] ✅ 일치: {info['predicted_action']}")
            else:
                label = info.get('step_number')
                if label is None:
                    label = info.get('index')
                print(f"  [Step {label}] ❌ 불일치:")
                print(f"    - 예측: {info['predicted_action']}")
                print(f"    - 정답: {info['expected_action']}")

            if done:
                print(f"\n에피소드 종료. 최종 보상: {reward}")
                break

        accuracy = (matched_steps / total_steps) * 100 if total_steps > 0 else 0
        print(f"\n--- 리플레이 결과 (Session {session_id}) ---")
        print(f"정확도: {accuracy:.2f}% ({matched_steps}/{total_steps} 스텝 일치)")
        print("-------------------------------------")


# --- 실행 예시 ---
if __name__ == '__main__':
    # 리플레이 하네스를 테스트하기 위한 간단한 실행 코드
    DEMO_FILE = 'webshop_demonstrations_0-100.json'

    # 1. 환경과 정책 초기화
    offline_env = OfflineWebshopEnv(DEMO_FILE)
    stub_policy = StubPolicy()

    # 2. 러너 초기화
    runner = ReplayRunner(env=offline_env, policy=stub_policy)

    # 3. 특정 세션(예: session_id 3) 리플레이 실행
    runner.run_episode(session_id=3)

    # 4. 다른 세션(예: session_id 9) 리플레이 실행
    runner.run_episode(session_id=9)
