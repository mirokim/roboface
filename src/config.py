"""Roboface 전역 설정 — 핀맵, 상수, 환경 변수.

simulator 모드와 robot 모드를 환경변수로 구분.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# .env 자동 로드 (있으면)
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

Mode = Literal["simulator", "robot"]

# === 모드 ===
MODE: Mode = os.getenv("ROBOFACE_MODE", "simulator")  # type: ignore[assignment]

# 비전 모델 — "detect" (SSD MobileNet, 객체 감지) 또는 "pose" (HigherHRNet, 자세 추정).
# pose: 손 흔들기를 손목 좌표로 직접 트래킹 — 훨씬 정확하지만 HigherHRNet 모델 필요
# detect: 기존. person 감지만, wave는 motion-based.
VISION_MODE = os.getenv("VISION_MODE", "pose")

# === 디스플레이 (LCD or Pygame 창) ===
# 2.4" LCD 네이티브는 240×320(세로)지만, 로봇은 90° 회전해서 320×240(가로)로 사용
# (Stack-chan 정통 방향)
DISPLAY_WIDTH = 320
DISPLAY_HEIGHT = 240
FPS = 30

# 그리기 선 두께 — 모든 얼굴 요소에 일관 적용
LINE_THICK = 4

# === 색상 (RGB) — 순수 모노톤 (검정 + 흰색) ===
COLOR_BG = (0, 0, 0)             # 검정
COLOR_EYE = (255, 255, 255)      # 흰색
COLOR_EYE_DARK = (60, 60, 60)    # 미사용
COLOR_MOUTH = (255, 255, 255)    # 흰색
COLOR_BLUSH = (200, 200, 200)    # 옅은 회색 (거의 안 보일 정도)
COLOR_INDICATOR_REC = (255, 100, 100)  # 녹음 인디케이터만 빨강

# === LCD 핀 (Pi 5 BCM) — robot 모드에서만 의미있음 ===
LCD_SPI_PORT = 0
LCD_SPI_DEVICE = 0       # CE0
LCD_DC_PIN = 25
LCD_RESET_PIN = 27
LCD_BACKLIGHT_PIN = 18

# === I2C (PCA9685 + MPU6050 미래용) ===
I2C_BUS = 1
PCA9685_ADDRESS = 0x40

# === 서보 (PCA9685 채널) ===
SERVO_PAN_CHANNEL = 0
SERVO_TILT_CHANNEL = 1

# SG92R 270° 사양
SERVO_PULSE_MIN_US = 500
SERVO_PULSE_MAX_US = 2500
SERVO_RANGE_DEG = 270

# 가동 범위 (소프트웨어 리밋)
PAN_MIN_DEG = 45     # 좌
PAN_CENTER_DEG = 135 # 정면
PAN_MAX_DEG = 225    # 우
TILT_MIN_DEG = 50    # 위
TILT_CENTER_DEG = 90
TILT_MAX_DEG = 130   # 아래 (5~85° 범위 안에서)

# === 센서 핀 ===
DHT22_GPIO = 22
MMWAVE_UART = "/dev/serial0"
MMWAVE_BAUDRATE = 115200

# === 음성 (STT/TTS/Wake word) ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PORCUPINE_ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY", "")
PORCUPINE_KEYWORD = os.getenv("PORCUPINE_KEYWORD", "jarvis")
PORCUPINE_KEYWORD_PATH = os.getenv("PORCUPINE_KEYWORD_PATH", "") or None
AUDIO_INPUT_DEVICE_RAW = os.getenv("AUDIO_INPUT_DEVICE", "")
# 숫자면 int, 문자열이면 그대로 (sounddevice가 이름 부분 매칭 지원)
AUDIO_INPUT_DEVICE: int | str | None
if AUDIO_INPUT_DEVICE_RAW == "":
    AUDIO_INPUT_DEVICE = None
elif AUDIO_INPUT_DEVICE_RAW.lstrip("-").isdigit():
    AUDIO_INPUT_DEVICE = int(AUDIO_INPUT_DEVICE_RAW)
else:
    AUDIO_INPUT_DEVICE = AUDIO_INPUT_DEVICE_RAW

# === Anthropic ===
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-haiku-4-5-20251001"  # 응답 빠르고 저렴
CLAUDE_MODEL_HEAVY = "claude-sonnet-4-6"    # 일정 추출 등 정밀 작업

# === ThinkTank 통합 ===
THINKTANK_BASE_URL = os.getenv("THINKTANK_BASE_URL", "http://localhost:3001")
THINKTANK_ROBOT_TOKEN = os.getenv("THINKTANK_ROBOT_TOKEN", "")
THINKTANK_TIMEOUT_SEC = 5
THINKTANK_RETRY = 3

# === 로깅 ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# === 데이터 경로 ===
DATA_DIR = ROOT / "src" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "roboface.db"

# === 행동 파라미터 ===
@dataclass(frozen=True)
class BehaviorConfig:
    """캐릭터 행동 튜닝 값."""

    # 깜빡임
    blink_min_interval_sec: float = 2.0
    blink_max_interval_sec: float = 6.0
    blink_duration_ms: int = 150

    # idle 두리번
    idle_look_min_interval_sec: float = 8.0
    idle_look_max_interval_sec: float = 20.0
    idle_look_duration_ms: int = 800

    # 작업시간
    work_break_warn_minutes: int = 60       # 1시간
    work_break_strong_minutes: int = 120    # 2시간
    work_break_alarm_minutes: int = 240     # 4시간

    # 자세
    posture_warn_continuous_sec: int = 600  # 10분
    posture_strong_continuous_sec: int = 1200  # 20분

    # 능동 멘트 빈도 제한
    proactive_min_silence_sec: int = 180    # 3분 침묵 후 OK
    proactive_max_per_hour: int = 12
    proactive_quiet_hours: tuple[int, int] = (22, 7)  # 밤 10시~아침 7시는 자제

    # 잡담 (chit-chat) — long_silence보다 가볍게, 사용자 있을 때 짧게 말 걸기
    chitchat_min_interval_sec: int = 240    # 마지막 발화 후 4분 지나면 후보
    chitchat_max_interval_sec: int = 600    # 10분 안엔 무조건 한 번

    # 절전
    idle_screen_dim_after_sec: int = 300    # 5분 부재
    idle_screen_off_after_sec: int = 1800   # 30분 부재


BEHAVIOR = BehaviorConfig()


def is_simulator() -> bool:
    return MODE == "simulator"


def is_robot() -> bool:
    return MODE == "robot"
