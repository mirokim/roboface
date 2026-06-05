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
SERVO_PAN_CHANNEL = 1    # 좌우 회전
SERVO_TILT_CHANNEL = 4   # 상하 회전 (채널 1은 PAN 점유, TILT는 4번)

# SG92R 270° 사양
SERVO_PULSE_MIN_US = 500
SERVO_PULSE_MAX_US = 2500
SERVO_RANGE_DEG = 270

# 가동 범위 (소프트웨어 리밋) — SG90 180° 안에서 안전 마진 두고
# 절대 0/180에 가까이 못 가게 (stall 방지)
PAN_MIN_DEG = 30     # 좌
PAN_CENTER_DEG = 90  # 정면
PAN_MAX_DEG = 150    # 우
TILT_MIN_DEG = 70    # 위
TILT_CENTER_DEG = 90
TILT_MAX_DEG = 110   # 아래

# === 센서 핀 ===
DHT22_GPIO = 22
MMWAVE_UART = "/dev/serial0"
MMWAVE_BAUDRATE = 115200

# === 음성 (STT/TTS/Wake word) ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# TTS 명시적 비활성 — 스피커 없는 셋업. OPENAI_API_KEY가 있어도 TTS는 X,
# fake animation(입모양만)로 폴백. STT는 영향 받지 않음.
TTS_DISABLED = os.getenv("TTS_DISABLED", "").lower() in ("1", "true", "yes")
# ambient 주변 청취 활성. 기본 False. AMBIENT_LISTEN=1 + OPENAI_API_KEY 있으면
# WhisperVADStreamer(진짜 STT) 사용. OPENAI_API_KEY 없으면 동작 X (mock는 의도적
# fallback 안 함 — 가짜 발화로 conversation_log 오염 방지).
AMBIENT_LISTEN = os.getenv("AMBIENT_LISTEN", "").lower() in ("1", "true", "yes")
# voice_assistant(wake word + 별도 STT) 명시적 비활성. AMBIENT_LISTEN이 always-on
# STT 모드라 wake word 없이 발화 다 잡힘 → voice_assistant와 중복.
WAKE_DISABLED = os.getenv("WAKE_DISABLED", "").lower() in ("1", "true", "yes")
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

# === Web UI ===
WEB_UI_PORT = int(os.getenv("WEB_UI_PORT", "8080"))
WEB_UI_PASSWORD = os.getenv("WEB_UI_PASSWORD", "")    # 빈 문자열 → UI 비활성

# === LLM 백엔드 선택 ===
# "claude" → Anthropic API (기본, 온라인 필요)
# "local"  → llama-cpp-python + GGUF 모델 (오프라인, 비용 0)
# 봇 setup이 모델 다운로드 + .env에 LLM_BACKEND=local 설정.
LLM_BACKEND = os.getenv("LLM_BACKEND", "claude").lower()

# === Anthropic ===
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# 모델 분리: idle 점검(do_nothing 위주)엔 Haiku, 사용자 발화/호명 직후엔 Sonnet.
# Haiku 4.5의 speak 빈 input 버그는 agent._do_speak가 empty text skip으로 가드.
# 사용자 응답 누락 위험은 Sonnet 사용 윈도우에서 0.
CLAUDE_MODEL = "claude-sonnet-4-6"          # 사용자 발화/호명 직후 — 응답 보장
CLAUDE_MODEL_LIGHT = "claude-haiku-4-5"     # idle tick — 대부분 do_nothing
CLAUDE_MODEL_HEAVY = "claude-sonnet-4-6"    # 일정 추출 등 정밀 작업

# === ThinkTank 통합 ===
THINKTANK_BASE_URL = os.getenv("THINKTANK_BASE_URL", "http://localhost:3001")
THINKTANK_ROBOT_TOKEN = os.getenv("THINKTANK_ROBOT_TOKEN", "")
THINKTANK_TIMEOUT_SEC = 5
THINKTANK_RETRY = 3

# === OpenWeather (선택) — agent prompt에 날씨 한 줄 주입 ===
# 키 비어 있으면 WeatherClient.snapshot()이 None 반환 → agent에서 자동 skip.
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
WEATHER_LAT = float(os.getenv("WEATHER_LAT", "37.3504"))   # 분당 default
WEATHER_LON = float(os.getenv("WEATHER_LON", "127.108"))
WEATHER_LOCATION_NAME = os.getenv("WEATHER_LOCATION_NAME", "분당")

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

    # 작업시간 (앉아있기) — 자주 일어나도록 임계값 낮춤
    work_break_gentle_minutes: int = 30     # 30분: 가볍게 환기
    work_break_warn_minutes: int = 45       # 45분: 권유
    work_break_strong_minutes: int = 90     # 90분: 강하게
    work_break_alarm_minutes: int = 180     # 3시간: 알람

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

    # 에이전트 결정 주기 — 30→60. 신호 변화(표정/시선/발화 등)는 별도
    # 변화 트리거로 즉시 응답하므로 base는 idle 점검용. 토큰 비용 ½.
    agent_interval_sec: float = 60.0        # Claude 결정 주기 (idle 점검)
    agent_speak_min_gap_sec: float = 90.0   # 에이전트 발화 사이 최소 간격
    agent_dance_min_gap_sec: float = 120.0  # 에이전트 dance 사이 최소 간격 (격렬 방지)

    # 에이전트 vision — 이미지 첨부해 더 풍부한 시각 컨텍스트 (비용 ↑).
    # True여도 매 tick 첨부 X. 다음 조건 중 하나 만족 시만:
    #   1) 직전 첨부 후 max_interval_sec 이상 경과
    #   2) 사용자 표정 변함 (current_emotion 전이)
    #   3) 활동성 또는 시선 타깃 전이
    #   4) PRESENCE_NEW 직후 (사람 새로 등장)
    agent_vision_enabled: bool = True
    agent_vision_min_interval_sec: float = 90.0    # 30→90. 비용 절감 (이미지 input가 가장 비쌈)
    agent_vision_max_interval_sec: float = 600.0   # 300→600. 10분에 한 번은 무조건 첨부
    agent_vision_jpeg_quality: int = 70            # 0~100. 70이면 320×240 ~10KB
    agent_vision_max_side_px: int = 480            # 큰 frame은 다운샘플 (비용 절감)

    # 인사
    greeting_cooldown_sec: float = 300.0    # 같은 사람에게 5분 안엔 다시 인사 X

    # 대화 기록
    history_recent_window_min: float = 60.0   # 에이전트가 보는 직전 대화 윈도우
    history_recent_turns: int = 20            # 에이전트가 보는 직전 턴 수
    history_voice_turns: int = 6              # voice_assistant 컨텍스트 턴 수

    # 발화 표시
    min_speech_display_sec: float = 10.0      # 말풍선 최소 노출 (마이크 없으니 충분히)
    speech_extra_hold_sec: float = 1.5        # 음성 끝나고 추가 노출

    # 입 모양 → 음량 (단일 SSOT — mouth.py + tts.py 공통)
    # 정규화된 0~1 진폭 기준. (small, mid, large) 임계값 이상이면 다음 단계.
    mouth_amp_thresholds: tuple[float, float, float] = (0.15, 0.35, 0.65)
    # raw RMS(마이크 입력) → 0~1 정규화 게인 (mouth.update_talking 전용)
    mouth_raw_rms_gain: float = 3.0

    # 폴링 주기
    proactive_eval_interval_sec: float = 1.0   # proactive_speaker.run_loop
    work_tracker_interval_sec: float = 60.0    # work_tracker.run
    sensor_poll_interval_sec: float = 0.1      # SensorManager.run

    # 날씨 (OpenWeather) — 매 tick 사용하지만 캐시로 API 호출 빈도 제한.
    # 1800s(30min): 무료 tier 1M calls/month 한참 안 깸 + 날씨 변화엔 충분히 빠름.
    weather_cache_sec: float = 1800.0
    weather_http_timeout_sec: float = 5.0
    # forecast(내일 등)는 변화 더 느림 → 1시간 캐시. 같은 endpoint 다른 데이터.
    weather_forecast_cache_sec: float = 3600.0


BEHAVIOR = BehaviorConfig()


def is_simulator() -> bool:
    return MODE == "simulator"


def is_robot() -> bool:
    return MODE == "robot"
