#!/bin/bash

# AgentQ Chrome 디버깅 모드 종료 스크립트

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

DEBUG_PORT=9222
USER_DATA_DIR="/tmp/chrome-debug-agentq"
PID_FILE="/tmp/chrome-debug-agentq.pid"

log_info "Chrome 디버깅 모드를 종료합니다..."

# PID 파일에서 프로세스 종료
if [ -f "$PID_FILE" ]; then
    CHROME_PID=$(cat "$PID_FILE")
    if kill -0 "$CHROME_PID" 2>/dev/null; then
        log_info "Chrome 프로세스 종료 중... (PID: $CHROME_PID)"
        kill "$CHROME_PID"
        sleep 2
        
        # 강제 종료가 필요한 경우
        if kill -0 "$CHROME_PID" 2>/dev/null; then
            log_warn "강제 종료 중..."
            kill -9 "$CHROME_PID"
        fi
    fi
    rm -f "$PID_FILE"
fi

# 포트 기반으로 프로세스 찾아서 종료
if pgrep -f "remote-debugging-port=$DEBUG_PORT" > /dev/null; then
    log_info "포트 기반으로 Chrome 프로세스 종료 중..."
    pkill -f "remote-debugging-port=$DEBUG_PORT"
    sleep 2
fi

# 사용자 데이터 디렉토리 정리
if [ -d "$USER_DATA_DIR" ]; then
    log_info "사용자 데이터 디렉토리 정리 중..."
    rm -rf "$USER_DATA_DIR"
fi

# 최종 확인
if ! pgrep -f "remote-debugging-port=$DEBUG_PORT" > /dev/null; then
    log_info "✅ Chrome 디버깅 모드가 성공적으로 종료되었습니다."
else
    log_error "❌ 일부 Chrome 프로세스가 여전히 실행 중입니다."
    log_info "수동으로 확인해주세요: ps aux | grep chrome"
fi