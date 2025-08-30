"""리플레이 및 디버깅 유틸리티"""
import json
from typing import Dict, Any, List
from datetime import datetime
from state import SearchState


class ReplayLogger:
    """검색 과정 로깅 및 리플레이"""
    
    def __init__(self, log_dir: str = "./runs"):
        self.log_dir = log_dir
        self.current_session = None
    
    def start_session(self, goal: str, config: Dict[str, Any] = None) -> str:
        """새 세션 시작"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"session_{timestamp}"
        
        self.current_session = {
            "session_id": session_id,
            "goal": goal,
            "config": config or {},
            "start_time": timestamp,
            "states": [],
            "actions": [],
            "scores": []
        }
        
        return session_id
    
    def log_state(self, state: SearchState, score: float):
        """상태 로깅"""
        if not self.current_session:
            return
        
        self.current_session["states"].append({
            "timestamp": datetime.now().isoformat(),
            "state": self._serialize_state(state),
            "score": score
        })
    
    def log_action(self, action: str, result: Dict[str, Any]):
        """액션 로깅"""
        if not self.current_session:
            return
        
        self.current_session["actions"].append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "result": result
        })
    
    def end_session(self, final_result: Dict[str, Any]):
        """세션 종료 및 저장"""
        if not self.current_session:
            return
        
        self.current_session["end_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session["final_result"] = final_result
        
        # 파일로 저장
        filename = f"{self.log_dir}/{self.current_session['session_id']}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.current_session, f, ensure_ascii=False, indent=2)
        
        session_id = self.current_session["session_id"]
        self.current_session = None
        return session_id
    
    def _serialize_state(self, state: SearchState) -> Dict[str, Any]:
        """상태를 직렬화 가능한 형태로 변환"""
        serialized = {}
        for key, value in state.items():
            if key == 'frontier':
                # 프론티어는 크기만 저장
                serialized[key] = len(value) if value else 0
            elif key == 'observation':
                # 관측 정보 요약
                obs = value
                serialized[key] = {
                    "query": obs.get('query'),
                    "page": obs.get('page'),
                    "sort": obs.get('sort'),
                    "filters": obs.get('filters'),
                    "result_count": len(obs.get('results', [])),
                    "cart_count": len(obs.get('cart', []))
                }
            else:
                serialized[key] = value
        
        return serialized


def load_replay_session(session_file: str) -> Dict[str, Any]:
    """리플레이 세션 로드"""
    with open(session_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_session(session_data: Dict[str, Any]) -> Dict[str, Any]:
    """세션 분석"""
    states = session_data.get("states", [])
    actions = session_data.get("actions", [])
    
    analysis = {
        "goal": session_data.get("goal"),
        "total_states": len(states),
        "total_actions": len(actions),
        "success": session_data.get("final_result", {}).get("success", False),
        "final_score": session_data.get("final_result", {}).get("best_score", 0),
        "score_progression": [s["score"] for s in states],
        "action_types": {}
    }
    
    # 액션 타입별 통계
    for action_log in actions:
        action_str = action_log["action"]
        action_type = action_str.split(":")[0] if ":" in action_str else action_str
        analysis["action_types"][action_type] = analysis["action_types"].get(action_type, 0) + 1
    
    return analysis