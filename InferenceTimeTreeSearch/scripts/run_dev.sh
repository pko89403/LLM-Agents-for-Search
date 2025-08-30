#!/bin/bash
# 개발 환경 실행 스크립트

echo "🚀 ITTS 에이전트 개발 환경 시작"

# 환경변수 로드
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Python 경로 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 메인 실행
python main.py