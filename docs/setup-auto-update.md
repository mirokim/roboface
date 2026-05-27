# 무선 자동 업데이트 셋업

봇에 SSH 한 번 들어가서 인스톨하면, 그 후로는:
- **자동**: `git push origin main` → 봇이 5분 안에 자동 pull + 재시작
- **즉시**: 같은 wifi에서 web UI → "⟳ 코드 업데이트" 버튼 클릭

## 1회 인스톨 (봇에서 SSH 들어가 한 번만)

```bash
cd ~/roboface
git pull   # auto_update.sh + systemd unit 파일 받기

# 1. 스크립트 실행 권한
chmod +x scripts/auto_update.sh

# 2. systemd unit 설치
sudo cp scripts/roboface-update.service /etc/systemd/system/
sudo cp scripts/roboface-update.timer   /etc/systemd/system/
sudo systemctl daemon-reload

# 3. timer 활성화 (5분마다 auto-update)
sudo systemctl enable --now roboface-update.timer

# 4. sudoers — auto_update.sh가 봇 서비스 재시작 + web UI가
#    update.service 실행할 수 있도록 NOPASSWD 허용.
sudo tee /etc/sudoers.d/roboface-update > /dev/null <<'EOF'
miro ALL=(ALL) NOPASSWD: /bin/systemctl restart roboface.service
miro ALL=(ALL) NOPASSWD: /bin/systemctl start roboface-update.service
EOF
sudo chmod 0440 /etc/sudoers.d/roboface-update
```

## 동작 확인

```bash
# timer 상태
systemctl status roboface-update.timer

# 다음 발동 시각
systemctl list-timers roboface-update.timer

# 마지막 실행 로그 (journalctl 또는 syslog의 'rfup' 태그)
journalctl -t rfup -n 30
```

## 즉시 트리거 (web UI 없이)

```bash
sudo systemctl start roboface-update.service
journalctl -t rfup -f   # 로그 라이브로
```

## 안전 장치

- **fetch 실패** (네트워크 끊김): 조용히 패스, 다음 tick에서 재시도
- **ff-only merge**: 봇에서 누군가 직접 commit해 origin/main과 분기됐으면 자동 merge X — 사람 손 필요
- **변경 없으면 재시작 X**: 매 5분마다 fetch만 함, pull/restart는 진짜 새 commit 있을 때만
- **requirements 변경 감지**: `requirements*.txt` 바뀌면 자동 `pip install`
- **재시작 실패해도**: 코드는 이미 pull됐지만 systemd가 alive 유지 — 다음 깃 push로 복구 가능

## 끄기

```bash
sudo systemctl disable --now roboface-update.timer
```

## 트러블슈팅

**"sudoers entry?" 로그 → restart 실패**
`/etc/sudoers.d/roboface-update`의 사용자명/경로 확인. `which systemctl`이 `/bin/systemctl`인지 `/usr/bin/systemctl`인지 OS 따라 다름.

**web UI 버튼이 "systemctl 실패"**
web UI 프로세스 user(보통 `miro`)에 `systemctl start roboface-update.service` sudoers 허용됐는지 확인. sudoers 줄에 `start`가 빠졌으면 추가.

**timer는 도는데 pull 안 됨**
`cd ~/roboface && git fetch origin main`을 손으로 실행해서 SSH key/HTTPS 인증이 작동하는지 확인. GitHub 공개 repo면 보통 OK.
