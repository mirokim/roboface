# 🌅 아침 보고서 (2026-05-15)

**작성 시각**: 2026-05-15 02:55 KST
**작업자**: 야간 자율 모드 (Claude)
**요청**: "나 자고올게 작업좀 해줘"

---

## 📋 했던 일

### 1. AI Camera 사람 감지 통합 ⭐

| 모듈 | 역할 |
|---|---|
| `src/vision/camera.py` | IMX500 picamera2 래퍼, 5fps 객체 감지 비동기 스트림 |
| `src/vision/person_detector.py` | 디바운스 + 히스테리시스 (3프레임 확정 → AWAY 5초) |
| `src/tasks/vision_task.py` | 카메라 스트림 → sensor manager 이벤트 큐 |

**효과**: 카메라가 mmWave와 **동등한 사용자 감지 채널**. 둘 다 `PRESENCE_NEW/LEFT` 이벤트 emit → work_tracker가 자동 통합.

bbox 크기 → 거리 추정 (대략 100/sqrt(area) cm).

### 2. mmWave 실제 패킷 파서

이전엔 데이터 들어오면 그냥 `presence=True`로 처리하던 stub.
지금은 **HLK-LD2410 / S3KM1110 표준 모드 프로토콜** 정확히 디코드:

```
헤더 F4F3F2F1 → 길이 → 페이로드(13B) → 꼬리 F8F7F6F5

페이로드 디코딩:
  target_state: 0=없음 / 1=움직임 / 2=정적 / 3=둘 다
  거리(cm), 에너지(0-100)
```

이벤트:
- `PRESENCE_NEW` (등장)
- `PRESENCE_LEFT` (이탈)
- `PRESENCE_STATIC` (정적 — 호흡 등 미세 감지)
- `DISTANCE_CHANGED` (3초마다 throttle)

**보드레이트 자동 감지**도 추가 (115200 → 256000 → 9600 순). 어제 데이터 안 들어왔던 원인이 보드레이트 차이일 수도 있어 자동 시도.

### 3. main_robot.py 통합

```python
asyncio.create_task(run_vision(lambda ev: sensors.events.append(ev)))
```

vision_task가 sensor manager의 큐에 직접 push → 기존 핸들러가 자동으로 처리.

### 4. 테스트 (29개 → 41개)

| 테스트 | 새로 추가 |
|---|---|
| `test_mmwave_parser.py` | 5개 (페이로드 디코딩 검증) |
| `test_person_detector.py` | 7개 (디바운스/히스테리시스/bbox 거리) |

**전체 41개 통과** ✅

---

## 🎯 일어났을 때 할 일

### A. Pull + 의존성 확인 (1분)

```bash
ssh miro@192.168.0.46
cd ~/roboface
git pull
source .venv/bin/activate
```

새 코드만 추가됐고 새 의존성 없음.

### B. mmWave 결선 재시도 (10분, 가장 중요)

어제 미해결. 보드레이트 자동 감지 추가했으니 결선만 정확히 되면 자동으로 잡힐 거예요.

```bash
sudo shutdown -h now
# 전원 분리 → 결선 확인
```

**T-Cobbler 사진 한 장 찍어서 확인하는 게 가장 빠름.**

| mmWave | T-Cobbler |
|---|---|
| 3V3 | `3V3` |
| GND | `GND` |
| **TX** | **GPIO15 (RXD)** ← 교차 |
| **RX** | **GPIO14 (TXD)** ← 교차 |
| OT2 | 비워둠 |

전원 인가 후 SSH 접속하면 main_robot에서 자동으로 보드레이트 감지하고 데이터 수신 시작합니다.

### C. 카메라 사람 감지 동작 확인 (10분)

mmWave 안 풀려도 카메라만으로 사람 감지 가능. main_robot 실행:

```bash
python -m src.main_robot
```

카메라 앞에 가면:
```
INFO  사람 감지 (카메라) — 거리 ~120cm
INFO  상태 전이: idle → watching
```

가면 5초 후:
```
INFO  사람 사라짐 (카메라)
INFO  상태 전이: watching → idle
```

⚠️ 첫 실행 시 IMX500 펌웨어 로딩에 약 30초 걸림 (정상).

### D. (선택) 받침대 / PCA9685 도착 대기

---

## ⚠️ 알려진 이슈

| 이슈 | 우회 |
|---|---|
| AI Camera는 `imx500-all` apt 패키지가 깔려야 동작 | 어제 설치 완료 ✅ |
| 카메라 모델 첫 로드 ~30초 | 초기 1회만, 그 뒤로는 빠름 |
| mmWave 보드레이트 다르면 자동 감지 0.5초씩 시도 | 정상 동작, 약간의 시작 지연 |
| 카메라 + mmWave 둘 다 PRESENCE 이벤트 emit | work_tracker가 중복 안 만들도록 처리 (현재 OK) |
| picamera2 의존성 무거움 (~수십 MB) | 이미 설치됨 |

---

## 📊 코드 통계

| 모듈 | 줄 수 | 비고 |
|---|---|---|
| `src/vision/camera.py` | 154 | 신규 |
| `src/vision/person_detector.py` | 96 | 신규 |
| `src/tasks/vision_task.py` | 42 | 신규 |
| `src/sensors/mmwave.py` | 215 | 대폭 개선 (84 → 215) |
| `tests/test_person_detector.py` | 72 | 신규 |
| `tests/test_mmwave_parser.py` | 51 | 신규 |
| `docs/2026-05-15_morning-report.md` | 본 문서 | 신규 |

**Total**: 7 파일, 약 +680 줄.

---

## 🧩 남은 큰 작업

- [ ] mmWave 결선 디버깅 (사진 보고 진단)
- [ ] PCA9685 도착 → 서보 첫 회전
- [ ] 받침대 (베이스 플레이트 + 댐퍼)
- [ ] 3D 프린팅 외피
- [ ] 오디오 통합 (USB 동글)
- [ ] LCD에서 `>_<` 표정 픽셀 확인 (smoothscale 적용된 상태)

---

푹 쉬셨길. 좋은 아침 ☀️
