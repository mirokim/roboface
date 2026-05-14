# Pi 5 초기 셋업 가이드

**대상**: 부품 도착 직후, Pi 5에 OS 설치 + 헤드리스 운영 준비.

---

## 1. SD 카드에 OS 굽기 (지금 단계)

### 1.1 Raspberry Pi Imager 실행

설치 완료 — 시작 메뉴에서 **"Raspberry Pi Imager"** 검색해서 실행.

### 1.2 디바이스 / OS / 저장소 선택

| 항목 | 선택 |
|---|---|
| **Device** | Raspberry Pi 5 |
| **OS** | Raspberry Pi OS (64-bit) — Bookworm, with desktop |
| **Storage** | H: (128GB SD) — 다른 USB 끼우지 않게 주의 |

### 1.3 ⭐ "설정 편집" (Ctrl+Shift+X 또는 다음 화면에서 "예")

**Roboface는 헤드리스(SSH) 운영**이라 미리 설정 필수.

**General 탭:**
- ✅ 호스트네임 설정: **`roboface`**
- ✅ 사용자 계정 설정:
  - 사용자 이름: **`miro`** (또는 원하는 이름)
  - 비밀번호: 본인 정함 (꼭 기록!)
- ✅ Wi-Fi 설정:
  - SSID: 본인 공유기 이름
  - 비밀번호: Wi-Fi 비번
  - Country: **`KR`**
- ✅ 로케일:
  - Timezone: **`Asia/Seoul`**
  - 키보드 레이아웃: **`us`** (또는 `kr`)

**Services 탭:**
- ✅ **SSH 사용 활성화** ← 필수!
- 인증: **비밀번호 사용** (또는 키 인증)

**Options 탭:**
- ✅ "Eject media when finished" 체크

### 1.4 쓰기 시작
- WRITE 버튼 → 약 5~10분 소요
- 검증까지 끝나면 자동 마운트 해제됨

---

## 2. Pi 5 첫 부팅

### 2.1 SD 카드 + 케이블 연결
1. SD 카드를 Pi 5에 삽입
2. **Argon Poly+ 5 케이스 조립** (Pi + 액티브쿨러 부착)
3. micro-HDMI 케이블 (선택, 헤드리스면 안 꽂아도 됨)
4. **27W USB-C 어댑터** 전원 연결

### 2.2 첫 부팅 (약 1분)
- 빨강 LED → 녹색 LED 깜빡임 (정상 부팅)
- Wi-Fi 자동 연결 (Imager에서 미리 설정한 것)
- SSH 자동 활성화

---

## 3. SSH로 접속

### 3.1 Pi IP 확인
공유기 관리 페이지에서 **`roboface`** 호스트 찾기, 또는:

```powershell
# Windows에서 ping (Bonjour/mDNS 작동 시)
ping roboface.local
# 또는
arp -a | findstr roboface
```

### 3.2 SSH 접속

```powershell
ssh miro@roboface.local
# 또는
ssh miro@192.168.0.XX
```

처음 접속 시 호스트 키 신뢰 묻는 거 `yes`.

---

## 4. 인터페이스 활성화 (SPI / I2C / UART)

Pi 5에 접속 후:

```bash
sudo raspi-config
```

메뉴:
- **3. Interface Options** →
  - **I4. SPI** → Enable ✅
  - **I5. I2C** → Enable ✅
  - **I6. Serial Port** →
    - "login shell over serial" → **No**
    - "serial port hardware enabled" → **Yes** ✅

종료 후 **재부팅** (`sudo reboot`).

---

## 5. Roboface 코드 배포

재부팅 후 SSH 재접속:

```bash
# 시스템 업데이트
sudo apt update && sudo apt full-upgrade -y

# Python 도구
sudo apt install -y python3-venv python3-pip git

# 저장소 clone
cd ~
git clone https://github.com/mirokim/roboface.git
cd roboface

# 가상환경 + 의존성
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-pi.txt
```

---

## 6. 환경 변수 설정

```bash
cp .env.example .env
nano .env
```

수정할 항목:
```
ROBOFACE_MODE=robot
ANTHROPIC_API_KEY=sk-ant-...
THINKTANK_BASE_URL=http://(thinktank 서버 IP):3001
THINKTANK_ROBOT_TOKEN=(thinktank의 ROBOT_SERVICE_TOKEN과 동일 값)
```

---

## 7. 첫 테스트 (LCD/서보 연결 전)

```bash
# 가상환경 활성화 상태에서
python -m src.main
```

LCD/서보 없어도:
- 로그가 정상 출력되는지
- mock 센서가 동작하는지

확인하고 Ctrl+C로 종료.

---

## 8. 다음 단계 (하드웨어 연결)

본격 통합 순서:

1. **PCA9685 + 서보** → I2C 연결 + 첫 회전 테스트
2. **2.4" LCD** → SPI 연결 + 첫 픽셀 표시
3. **AI Camera** → CSI 연결 + 영상 확인
4. **mmWave** → UART 연결 + 패킷 수신
5. **DHT22** → GPIO 22 연결 + 온도 출력
6. **오디오** (USB) → 마이크/스피커 테스트

각 단계는 별도 가이드 (도착 후 작성).

---

## 트러블슈팅

| 문제 | 해결 |
|---|---|
| SSH 접속 안 됨 | Wi-Fi 설정 / 호스트네임 확인. 공유기에서 IP 직접 찾기 |
| `pip install` 실패 | `sudo apt install python3-dev libffi-dev libssl-dev` 먼저 |
| pygame 설치 실패 | Pi에선 시뮬레이터 불필요 — `requirements-pi.txt`만 사용 |
| 부팅 안 됨 | SD 카드 다시 굽기. Imager가 검증 실패했을 수도 |
