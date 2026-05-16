#!/bin/bash
# Roboface 자동 시작 설치 — Pi에서 한 번만 실행
# 부팅 시 main_robot 자동 시작 + 죽으면 재시작
set -e

SERVICE_NAME="roboface.service"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "📁 프로젝트 경로: $PROJECT_DIR"

# 1. 로그 디렉토리 생성
mkdir -p "$PROJECT_DIR/logs"
echo "✓ logs/ 디렉토리 준비"

# 2. service 파일 복사 (경로가 다르면 sed로 치환)
TMPFILE=$(mktemp)
sed "s|/home/miro/roboface|$PROJECT_DIR|g; s|User=miro|User=$USER|g; s|Group=miro|Group=$USER|g" \
    "$SCRIPT_DIR/$SERVICE_NAME" > "$TMPFILE"

sudo cp "$TMPFILE" "/etc/systemd/system/$SERVICE_NAME"
rm -f "$TMPFILE"
echo "✓ /etc/systemd/system/$SERVICE_NAME 설치"

# 3. systemd 리로드 + 활성화
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
echo "✓ 부팅 시 자동 시작 등록"

# 4. 지금 시작
sudo systemctl restart "$SERVICE_NAME"
sleep 2
echo ""
echo "📊 현재 상태:"
sudo systemctl status "$SERVICE_NAME" --no-pager || true

echo ""
echo "🎉 설치 완료!"
echo ""
echo "관리 명령:"
echo "  sudo systemctl status roboface     # 상태 보기"
echo "  sudo systemctl stop roboface       # 정지"
echo "  sudo systemctl start roboface      # 시작"
echo "  sudo systemctl restart roboface    # 재시작"
echo "  sudo systemctl disable roboface    # 자동 시작 해제"
echo ""
echo "로그 보기:"
echo "  tail -f $PROJECT_DIR/logs/roboface.log"
echo "  journalctl -u roboface -f          # systemd 로그"
