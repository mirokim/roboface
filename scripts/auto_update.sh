#!/usr/bin/env bash
# Roboface 무선 자동 업데이트 — git fetch + ff-only pull + 의존성 + 재시작.
#
# 호출 경로:
#   - systemd timer (roboface-update.timer)가 5분마다
#   - 또는 web UI의 /api/update 버튼 (systemctl start roboface-update.service)
#
# 안전 장치:
#   - fetch 실패(네트워크) → 조용히 종료, 다음 tick에서 재시도
#   - ff-only merge — 로컬 충돌이면 merge 안 함 (사람 손 필요)
#   - 변경 없으면 재시작 X
#   - 재시작 실패해도 이전 코드는 그대로 (이미 pull 된 후라 약간 위험은 있음)

set -u

REPO_DIR="${ROBOFACE_DIR:-/home/miro/roboface}"
SERVICE_NAME="${ROBOFACE_SERVICE:-roboface.service}"
VENV_PIP="${ROBOFACE_VENV:-$REPO_DIR/.venv}/bin/pip"
LOG_TAG="rfup"

log() { logger -t "$LOG_TAG" "$1"; echo "[$LOG_TAG] $1"; }

cd "$REPO_DIR" 2>/dev/null || { log "repo not found: $REPO_DIR"; exit 1; }

CURRENT=$(git rev-parse HEAD 2>/dev/null) || { log "git rev-parse failed"; exit 1; }

# fetch 실패는 네트워크 문제 — 조용히 패스 (다음 tick에서)
if ! git fetch origin main --quiet 2>/dev/null; then
    log "fetch failed (network?)"
    exit 0
fi

REMOTE=$(git rev-parse origin/main 2>/dev/null) || { log "remote rev-parse failed"; exit 1; }

if [ "$CURRENT" = "$REMOTE" ]; then
    exit 0   # 변경 없음 — 빠르게 종료
fi

log "update available: ${CURRENT:0:7} -> ${REMOTE:0:7}"

# ff-only — 로컬 변경(unmerged/divergent)이면 자동 적용 안 함
if ! git merge --ff-only origin/main 2>&1 | logger -t "$LOG_TAG"; then
    log "ff merge failed — local divergence? skip restart"
    exit 1
fi

# 의존성 변경 시 pip install
if git diff --name-only "$CURRENT" "$REMOTE" 2>/dev/null \
        | grep -qE '^requirements(-pi)?\.txt$'; then
    log "requirements changed — pip install"
    if [ -f requirements-pi.txt ] && [ -x "$VENV_PIP" ]; then
        "$VENV_PIP" install -r requirements-pi.txt --quiet 2>&1 \
            | logger -t "$LOG_TAG"
    fi
fi

log "restart $SERVICE_NAME"
if sudo -n systemctl restart "$SERVICE_NAME" 2>&1 | logger -t "$LOG_TAG"; then
    log "update complete"
else
    log "restart failed — sudoers entry?"
    exit 1
fi
