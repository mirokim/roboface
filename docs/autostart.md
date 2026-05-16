# 자동 시작 (Auto-start) 설정

Pi 5 전원 켤 때 SSH 접속 없이 Roboface 자동 실행.

## 1회 설치

Pi에서 SSH 접속 후:

```bash
cd ~/roboface
git pull
bash scripts/install-autostart.sh
```

설치 스크립트가:
1. `logs/` 디렉토리 생성
2. `/etc/systemd/system/roboface.service` 설치
3. 부팅 시 자동 시작 등록
4. 즉시 시작
5. 상태 출력

## 동작 확인

```bash
# 현재 상태
sudo systemctl status roboface

# 출력 예시:
# ● roboface.service - Roboface AI Companion Robot
#      Loaded: loaded (/etc/systemd/system/roboface.service; enabled)
#      Active: active (running) since ...
#      ...
```

`Active: active (running)` 보이면 동작 중.

## 관리 명령

| 명령 | 동작 |
|---|---|
| `sudo systemctl status roboface` | 상태 |
| `sudo systemctl stop roboface` | 정지 |
| `sudo systemctl start roboface` | 시작 |
| `sudo systemctl restart roboface` | 재시작 |
| `sudo systemctl disable roboface` | 자동 시작 해제 |
| `sudo systemctl enable roboface` | 자동 시작 등록 |

## 로그 보기

```bash
# 파일 로그
tail -f ~/roboface/logs/roboface.log

# systemd 로그 (전체)
journalctl -u roboface -f

# 마지막 100줄
journalctl -u roboface -n 100
```

## 코드 업데이트 후

```bash
cd ~/roboface
git pull
sudo systemctl restart roboface
```

## 트러블슈팅

### 시작 안 됨
```bash
sudo systemctl status roboface
# Error 메시지 확인
journalctl -u roboface -n 50
```

흔한 원인:
- `.env` 파일 없음 → `cp .env.example .env` 후 편집
- venv 경로 잘못 → service 파일 `ExecStart` 경로 확인
- 권한 문제 → `chown -R miro:miro ~/roboface`

### 시작은 되는데 즉시 죽음
```bash
tail -100 ~/roboface/logs/roboface.log
```

LCD/카메라/PCA9685 초기화 실패 시에도 service는 살아있어야 함 (graceful fallback). 만약 진짜 죽으면 `RestartSec=10` 후 재시작.

### Restart 무한 루프
의존성 (예: 카메라) 미연결로 계속 죽으면 정지하고 디버깅:
```bash
sudo systemctl stop roboface
cd ~/roboface && source .venv/bin/activate
python -m src.main_robot   # 직접 실행해서 에러 확인
```

## 부팅 시간

전원 인가 후 약 **30~60초** 이내에:
- OS 부팅
- 네트워크 연결
- roboface 서비스 시작
- IMX500 펌웨어 로딩 (~5초)
- LCD에 첫 표정

## 종료

전원 분리 전 안전 종료:
```bash
sudo shutdown -h now
```

또는 service만 정지:
```bash
sudo systemctl stop roboface
```
