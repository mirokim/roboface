# Roboface

Stack-chan 스타일 데스크탑 AI 동반자 로봇.

> 🤖 사용자를 곁에서 조용히 관찰하다가 적절한 순간에만 입을 여는 캐릭터

**하드웨어**: Raspberry Pi 5 (8GB) + 2.4" LCD + 서보 2개 + AI Camera + mmWave + DHT22
**소프트웨어**: Python 3.11+ / Pygame / Anthropic Claude API
**통합**: [ThinkTank1](../thinktank1) (Life OS) — 자동 저널/일정/Vault RAG

자세한 설계는:
- [docs/parts-list.md](docs/parts-list.md) — 부품 리스트
- [docs/2026-05-10_1742_architecture-plan.md](docs/2026-05-10_1742_architecture-plan.md) — 전체 아키텍처

---

## 빠른 시작 (시뮬레이터 모드)

부품이 도착하기 전에도 PC에서 얼굴 시뮬레이터를 돌릴 수 있어요.

### 1. 의존성 설치

```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 환경 변수 (선택)

```bash
copy .env.example .env
# .env 열어서 ANTHROPIC_API_KEY 등 채우기 (없어도 mock 모드로 동작)
```

### 3. 실행

```bash
python -m src.main
```

### 키보드 단축키

| 키 | 동작 |
|---|---|
| `1` ~ `9`, `0` | 표정 전환 (NEUTRAL / HAPPY / EXCITED / SLEEPY / SURPRISED / WORRIED / FOCUSED / LOVE / THINKING / WINK) |
| `B` | 강제 깜빡임 |
| `SPACE` | mmWave 사용자 등장 트리거 (시뮬레이션) |
| `Q` / `ESC` | 종료 |

---

## 실제 로봇에서 (Pi 5)

부품 도착 후 라즈베리파이에서:

```bash
# Pi 5 셋업 후
git clone <repo> roboface
cd roboface
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-pi.txt

# 환경 변수
cp .env.example .env
# ROBOFACE_MODE=robot 으로 변경
nano .env

# I2C / SPI / UART 활성화 (raspi-config)
sudo raspi-config

python -m src.main
```

---

## 프로젝트 구조

```
roboface/
├── docs/                    # 설계 문서
│   ├── parts-list.md
│   └── 2026-05-10_1742_architecture-plan.md
├── src/
│   ├── config.py            # 핀맵, 상수, 모드 분기
│   ├── main.py              # 메인 진입점
│   │
│   ├── face/                # 얼굴 렌더링
│   │   ├── expressions.py   # 12종 표정 프리셋
│   │   ├── eyes.py          # 눈 그리기 + 깜빡임
│   │   ├── mouth.py         # 입 그리기 + 립싱크
│   │   └── renderer.py      # Pygame 렌더러
│   │
│   ├── motion/              # 머리 움직임
│   │   ├── servos.py        # PCA9685 / Mock
│   │   └── poses.py         # 인사/끄덕/두리번
│   │
│   ├── sensors/             # 입력 센서
│   │   ├── base.py
│   │   ├── mmwave.py        # S3KM1110 / Mock
│   │   ├── environment.py   # DHT22 / Mock
│   │   └── manager.py
│   │
│   ├── brain/               # 두뇌
│   │   ├── state_machine.py
│   │   ├── triggers.py      # 능동 개입 조건
│   │   ├── conversation.py  # Claude API
│   │   └── memory.py        # SQLite 저장
│   │
│   ├── integrations/
│   │   └── thinktank/       # ThinkTank1 연동
│   │       └── client.py
│   │
│   └── utils/
│       └── logger.py
│
├── tests/
├── requirements.txt         # 공통 의존성
├── requirements-pi.txt      # Pi 5 전용
└── .env.example
```

---

## 모드

| 모드 | 환경변수 | 설명 |
|---|---|---|
| `simulator` (기본) | `ROBOFACE_MODE=simulator` | PC에서 Pygame으로 얼굴만 표시. Mock 센서/서보. |
| `robot` | `ROBOFACE_MODE=robot` | Pi 5에서 실제 하드웨어 사용. |

모든 하드웨어 모듈(`sensors/`, `motion/`)은 같은 인터페이스로 두 모드 모두 동작.
**simulator로 작성한 코드를 그대로 robot에서 돌릴 수 있게** 설계됨.

---

## 핵심 기능 (Phase별)

| Phase | 내용 | 상태 |
|---|---|---|
| **1** | 얼굴 시뮬레이터 + 표정 + 깜빡임 | ✅ |
| **2** | 서보 추상화 + 센서 매니저 + 상태 머신 | ✅ |
| **3** | LCD 이식 + 실제 서보 연결 | ⏳ 부품 도착 후 |
| **4** | AI Camera + 자세 인식 | ⏳ |
| **5** | 음성 (STT/TTS) + Claude 대화 | ⏳ |
| **6** | ThinkTank 통합 (저널/일정/Vault RAG) | ⏳ |
| **7** | 능동 행동 + 패턴 학습 | ⏳ |

---

## 라이선스

개인 프로젝트. 비공개.
