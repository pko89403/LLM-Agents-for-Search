#!/bin/bash

# AgentQ Chrome 디버깅 모드 설정 스크립트

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

# Chrome 경로 확인
CHROME_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
DEBUG_PORT=9222
USER_DATA_DIR="/tmp/chrome-debug-agentq"

# Chrome 설치 확인
if [ ! -f "$CHROME_PATH" ]; then
    log_error "Chrome이 설치되지 않았습니다: $CHROME_PATH"
    log_info "Chrome을 설치하거나 경로를 확인해주세요."
    exit 1
fi

# 기존 Chrome 프로세스 확인 및 종료
log_info "기존 Chrome 디버깅 프로세스 확인 중..."
if pgrep -f "remote-debugging-port=$DEBUG_PORT" > /dev/null; then
    log_warn "기존 Chrome 디버깅 프로세스를 종료합니다..."
    pkill -f "remote-debugging-port=$DEBUG_PORT"
    sleep 2
fi

# 사용자 데이터 디렉토리 정리
if [ -d "$USER_DATA_DIR" ]; then
    log_info "기존 사용자 데이터 디렉토리를 정리합니다..."
    rm -rf "$USER_DATA_DIR"
fi

# 디렉토리 생성
mkdir -p "$USER_DATA_DIR"

log_info "Chrome 디버깅 모드를 시작합니다..."
log_info "포트: $DEBUG_PORT"
log_info "사용자 데이터 디렉토리: $USER_DATA_DIR"

# Chrome 디버깅 모드 실행
"$CHROME_PATH" \
    --remote-debugging-port=$DEBUG_PORT \
    --user-data-dir="$USER_DATA_DIR" \
    --no-first-run \
    --no-default-browser-check \
    --disable-extensions \
    --disable-plugins \
    --disable-default-apps \
    --disable-background-timer-throttling \
    --disable-backgrounding-occluded-windows \
    --disable-renderer-backgrounding \
    --disable-features=TranslateUI \
    --disable-ipc-flooding-protection \
    --no-sandbox \
    --disable-web-security \
    --disable-features=VizDisplayCompositor \
    > /dev/null 2>&1 &

CHROME_PID=$!

# Chrome 시작 대기
log_info "Chrome 시작을 기다리는 중..."
sleep 3

# 연결 테스트
if curl -s "http://localhost:$DEBUG_PORT/json/version" > /dev/null; then
    log_info "✅ Chrome 디버깅 모드가 성공적으로 시작되었습니다!"
    log_info "🌐 DevTools: http://localhost:$DEBUG_PORT"
    log_info "📊 Version Info: http://localhost:$DEBUG_PORT/json/version"
    log_info "🔧 Chrome PID: $CHROME_PID"
    
    # PID 파일 저장
    echo $CHROME_PID > /tmp/chrome-debug-agentq.pid
    log_info "💾 PID 파일 저장됨: /tmp/chrome-debug-agentq.pid"
else
    log_error "❌ Chrome 디버깅 모드 시작에 실패했습니다."
    exit 1
fi

log_info "🎯 AgentQ에서 사용할 준비가 완료되었습니다!"