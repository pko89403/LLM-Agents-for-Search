# LASER Post-Implementation Guide (Agent + Replay)

이 문서는 LangGraph 기반 LASER 에이전트와 리플레이(offline replay) 하네스를 구현/연동한 뒤, 무엇을 어떻게 진행하면 좋은지에 대한 실무용 체크리스트와 절차를 제공합니다. 아래 문서들과 함께 보시면 흐름이 명확해집니다.

관련 문서
- laser_agent_dev_guide.md: LangGraph 설계/구현 가이드
- offline_replay_harness.md: 오프라인 리플레이 하네스 설계/개발 가이드
- docs/laser_prompt_reference.md: 상태별 툴/프롬프트/매핑 레퍼런스

## 1) 빠른 체크리스트
- [ ] 에이전트 라이브 실행 검증: 샘플 세션 1~3개를 LangGraph 그래프로 직접 실행(graph.invoke/stream)
- [ ] 리플레이 데이터 준비: 에피소드 스냅샷(JSON) 포맷 결정 및 샘플 생성
- [ ] 오프라인 리플레이 검증: 리플레이 하네스로 동일 궤적 재현(graph.stream 기반 비교)
- [ ] 평가 메트릭 산출: 성공률, 유효 액션율, 평균 스텝, 비용/시간
- [ ] 베이스라인 스냅샷 저장: “known-good” 리플레이/결과를 아티팩트로 고정
- [ ] 리그레션 테스트: 베이스라인 대비 판정(허용 편차 설정)
- [ ] CI 파이프라인: 리플레이 + 평가 + 리그레션 자동화(선택적으로 요약 리포트 업로드)
- [ ] 운영 가드레일: 최대 스텝, 타임아웃, 구매 액션 인간검증 등 안전장치 설정

## 2) 라이브 실행 검증(Agent)
LangGraph 그래프를 간단 검증합니다.

예시(개요)
```python
from laser_langgraph_agent import build_graph, EnvHandle

envh = EnvHandle(observation_mode="html")
obs, url = envh.reset()

graph = build_graph()
init_state = {
  "user_instruction": "Find a budget wireless mouse under $20.",
  "obs": obs,
  "url": url,
  "current_laser_state": "Search",
  "step_count": 0,
  "_env": envh,
}

# 1) 스트리밍으로 단계별 상태 변화 확인
for event in graph.stream(init_state):
    print(event)

# 2) 최종 호출
final_state = graph.invoke(init_state)
print("FINAL:", final_state.get("selected_item"))
```

팁
- event 로깅을 파일로 저장해 디버깅에 활용하세요(JSON lines 권장).
- step_count, action_history, memory_buffer를 함께 출력하면 추적이 쉬워집니다.

## 3) 리플레이 데이터 준비
오프라인 재현을 위해 에피소드 스냅샷(JSON)을 준비합니다. 최소 포함 권장 필드:
- metadata: { agent_version, seed, timestamp, env_config }
- instruction: str
- steps: 배열
  - 각 step: { observation_before, allowed_actions, llm_prompt, llm_thought, llm_action, action_executed, observation_after, reward, done, timestamp }
- final: { reward, success, completed_by_backup }

데이터 출처
- 라이브 실행 시 graph.stream 이벤트를 수집하여 스냅샷화
- 기존 generate_demonstrations.py의 trajectory 구조를 참고해 변환

## 4) 오프라인 리플레이 실행
offline_replay_harness.md의 스펙을 구현한 래퍼를 통해, 동일한 입력/상태 델타를 순차 적용하고 결과를 비교합니다.

절차(개요)
1) 스냅샷 로드 → 초기 상태 구성(init_state)
2) 그래프 실행을 “재생 모드”로 설정: 노드의 정책/LLM 호출을 생략(또는 mock)하고, 저장된 action_executed를 그대로 env.step에 전달
3) 각 step마다 관찰/보상/종료 플래그를 비교(허용 오차 범위 내 판정)
4) 최종 보상/성공률 일치 여부 확인

성공 조건 예시
- 모든 step에서 observation_after가 해시(또는 key subset) 기준으로 일치
- 최종 reward/성공 플래그 일치

## 5) 평가와 리포팅
필수 지표
- 성공률(SR): 성공 에피소드 수 / 전체 에피소드 수
- 유효 액션율(VAR): 허용 액션 집합 내에서의 선택 비율(100% 권장)
- 평균 스텝 수(AL): 최종 상태 도달까지의 평균 단계 수
- 비용/속도: 모델 호출 횟수, 시간, 비용(선택)

리포트 출력
- 콘솔 요약 + JSON/CSV 저장
- 그래프(stream) 이벤트에서 주요 키(state deltas)만 추출한 “슬림 로그” 생성

## 6) 베이스라인 스냅샷과 리그레션 테스트
- 베이스라인 고정: 가장 최근의 “좋은 결과” 리플레이 스냅샷/지표를 artifacts로 저장
- 리그레션 기준: 성공률/평균 스텝/유효 액션율의 허용 편차 정의(예: SR -1% 이상, AL +0.5 이하)
- 테스트 실행: 신버전 → 리플레이/평가 → 베이스라인과 비교 → Pass/Fail

## 7) CI 파이프라인 권장 흐름
- step 1: 의존 설치, 환경 준비
- step 2: 소량 샘플(예: 10에피소드)로 리플레이 + 평가
- step 3: 지표 비교 → 실패 시 PR 차단
- step 4: 아티팩트 업로드(요약 리포트, 슬림 로그)

## 8) 운영 가드레일과 안전장치
- MAX_STEPS, 라우팅 기본값(router_fn), 타임아웃/재시도
- 구매 액션 보호: 실제 구매 전 인간검증/샌드박스 환경
- 체크포인트 저장소: MemorySaver → Redis/SQLite 전환(복원/관측/취소 용이)

## 9) 문제해결 가이드(요약)
- 관찰 불일치: 환경 버전/데이터 인덱스/시드 차이 확인, text_rich/HTML 모드 일치화
- 액션 라벨 미스매치: `[button] ... [button_]`between 텍스트 정규화 규칙 재확인
- 루프 고착: router_fn 개선, Next/Prev/Back 전이 조건 변경, score/heuristic 도입
- goals 비어있음/IndexError: env 초기화 로깅확인, 세션 인덱스 모듈러 매핑 적용(이미 패치됨)

## 10) 샘플 커맨드/워크플로우
- 라이브 데모(5개)
  - `python laser_agent.py --num_examples 5 --start 0`
- 데모 생성(리플레이 자료)
  - `python generate_demonstrations.py --num_examples 5 --start 0`
- LangGraph 기반 스켈레톤 테스트(예제 파일을 추가했다면)
  - `python laser_langgraph_agent.py`

## 11) 마무리
- 리플레이와 에이전트를 구현했다면, “재현성(리플레이) → 평가(지표) → 리그레션(CI)”의 고리를 만드는 것이 안정적인 개선의 핵심입니다. 위 체크리스트를 팀의 개발 흐름에 통합해 지속적으로 품질을 관리하세요.
