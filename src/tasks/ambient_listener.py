"""주변 청취 — STT로 사용자 발화 텍스트화 + 후속 처리.

스트리머 종류:
- MockSTT — 사전정의 문장 가끔 emit (개발/시뮬용)
- WhisperVADStreamer — 마이크 + VAD로 발화 잘라 OpenAI Whisper 호출

후속 처리:
- conversation_log "ambient" 기록 → agent 컨텍스트에 자동 노출
- 일정/약속 언급 → schedule_extractor로 전달
- 의미 있는 발화 → journal_writer로 전달
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any

from src.brain import memory
from src.utils.logger import get_logger

log = get_logger("ambient")


# Whisper(로컬 + cloud) 공통 hallucination 패턴 — 약한 신호/짧은 음에서
# 학습 데이터 잔재로 떨어지는 한국어 인사·유튜브 표현. ambient_listener에서
# 백엔드 무관 한 곳에서 필터.
_HALLUCINATION_PATTERNS = (
    "구독", "좋아요", "시청해", "시청 해", "시청해 주셔",
    "감사합니다", "고맙습니다", "고마워요", "고마워", "감사해",
    "수고하셨습니다", "수고하셨어요", "수고하세요",
    "안녕히 계세요", "안녕히 가세요",
    "MBC 뉴스", "KBS 뉴스", "한국어 자막", "자막 by",
    "이 영상", "다음 영상", "다음 시간",
)


def _is_hallucination(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    return any(pat in t for pat in _HALLUCINATION_PATTERNS)


# 로봇 이름 (캐릭터). 호명 패턴 매칭에 사용 — agent _AGENT_SYSTEM에도 동일.
ROBOT_NAME = "네모"


def _is_calling_robot(text: str) -> bool:
    """사용자가 로봇 호명한 발화인지 — 호격 또는 단어 단위 시작/끝 위치.

    matched:
      - "네모야 안녕" / "야 네모" / "네모아" (호격)
      - "네모 뭐해" / "안녕 네모" (단어 경계 시작/끝)
    not matched (false positive 차단):
      - "네모난 박스" (시작이지만 "네모"가 단어 아님 — 형용사 prefix)
      - "그 네모 영화" / "이거 네모 모양" (중간 위치)
      - "안녕 로봇" (이름 자체 X)
    """
    n_no_space = "".join(text.lower().split()).rstrip(".!?,~")
    if not n_no_space:
        return False
    # 호격 — 공백 무관, 가장 명확한 호명 ("네모야", "야네모")
    if f"{ROBOT_NAME}야" in n_no_space or f"{ROBOT_NAME}아" in n_no_space:
        return True
    if f"야{ROBOT_NAME}" in n_no_space:
        return True
    # 단어 단위 시작/끝 — 원본 token 단위 (단어 경계 보존).
    # "네모난"은 "네모"로 시작하지만 separate token이라 첫 토큰 != "네모" → 매칭 X.
    tokens = [t.lower().strip(".!?,~") for t in text.split()]
    tokens = [t for t in tokens if t]   # 빈 토큰 제거
    if not tokens:
        return False
    if tokens[0] == ROBOT_NAME or tokens[-1] == ROBOT_NAME:
        return True
    return False


def _wav_peak(wav_bytes: bytes) -> int:
    """WAV 바이트의 PCM peak 절댓값. 빠른 신호 강도 가드용."""
    import io
    import wave
    import numpy as np
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            pcm = wf.readframes(wf.getnframes())
        if not pcm:
            return 0
        return int(np.max(np.abs(np.frombuffer(pcm, dtype=np.int16))))
    except Exception:
        return 0


def _normalize_wav_peak(
    wav_bytes: bytes, target_peak: int = 28000, max_gain: float = 50.0,
    noise_floor: int = 80,
) -> bytes:
    """16-bit mono WAV peak를 target_peak로 정규화 (소프트웨어 게인).

    - peak가 noise_floor보다 작으면 무음 — 게인 안 줌 (잡음 증폭 방지)
    - max_gain 캡 — 노이즈 floor 너무 부풀리지 않게 상한
    """
    import io
    import wave
    import numpy as np

    in_buf = io.BytesIO(wav_bytes)
    with wave.open(in_buf, "rb") as wf:
        params = wf.getparams()
        pcm = wf.readframes(wf.getnframes())
    if not pcm:
        return wav_bytes
    arr = np.frombuffer(pcm, dtype=np.int16)
    peak = int(np.max(np.abs(arr)))
    if peak < noise_floor:
        return wav_bytes
    gain = min(max_gain, target_peak / peak)
    amplified = np.clip(
        arr.astype(np.float32) * gain, -32768, 32767,
    ).astype(np.int16)
    out_buf = io.BytesIO()
    with wave.open(out_buf, "wb") as wf:
        wf.setparams(params)
        wf.writeframes(amplified.tobytes())
    return out_buf.getvalue()


# Mock transcript pool — 부품 도착 전 시뮬레이션용
_MOCK_TRANSCRIPTS = [
    "내일 오후 3시에 김 부장님과 회의가 있어",
    "다음주 월요일까지 보고서 제출해야 해",
    "오늘은 좀 피곤하네",
    "점심은 뭐 먹지",
    "이번주 금요일에 친구랑 저녁 약속 잡았어",
    "프로젝트 마감이 다음달 15일이야",
    "이거 정말 재미있는데",
    "내일 9시에 치과 예약 있어",
    "주말에 영화 보러 갈 거야",
    "방금 그 회의 어땠어?",
]


class MockSTT:
    """가짜 STT — 가끔 무작위 transcript 생성."""

    def __init__(self, mean_interval_sec: float = 90.0) -> None:
        self.mean_interval = mean_interval_sec

    async def stream(self) -> AsyncIterator[str]:
        while True:
            wait = random.expovariate(1.0 / self.mean_interval)
            await asyncio.sleep(wait)
            text = random.choice(_MOCK_TRANSCRIPTS)
            log.info(f"[mock STT] \"{text}\"")
            yield text


class WhisperVADStreamer:
    """VAD always-on + OpenAI Whisper — wake word 없이 발화 → 텍스트.

    cost 보호:
    - perception.user_present False면 record skip (사람 없는데 STT X)
    - 너무 짧은 텍스트(2자 이하) 또는 너무 긴 utterance 제외
    - VADRecorder가 start_timeout 안에 발화 없으면 None 반환 — 그냥 다음 라운드

    시각 피드백 (face 주입 시):
    - VAD 발화 감지 → face.recording = True (LCD 우상단 빨간 dot 깜빡)
    - 발화 끝 → face.recording = False
    (이전엔 transcript echo도 LCD에 띄웠지만 거슬려서 제거 — recording dot만 유지)
    """

    def __init__(
        self,
        mic: Any,
        stt: Any,
        perception: Any | None = None,
        face: Any | None = None,
        max_sec: float = 15.0,
        silence_ms: int = 700,
        aggressiveness: int = 2,
    ) -> None:
        from src.audio.mic import VADRecorder
        self.mic = mic
        self.stt = stt
        self.perception = perception
        self.face = face
        self.recorder = VADRecorder(
            mic, aggressiveness=aggressiveness,
            silence_ms=silence_ms,
            start_timeout_sec=2.0,  # 짧게 — 사람 없으면 빨리 빠져나와 가드 재체크
            on_speech_start=self._on_speech_start,
            on_speech_end=self._on_speech_end,
        )
        self.max_sec = max_sec

    def _on_speech_start(self) -> None:
        if self.face is not None:
            self.face.recording = True

    def _on_speech_end(self) -> None:
        if self.face is not None:
            self.face.recording = False

    def _user_present(self) -> bool:
        if self.perception is None:
            return True
        return bool(getattr(self.perception, "user_present", True))

    # 마이크 callback이 이 시간(초) 이상 멈추면 ALSA broken으로 보고 재시작.
    _MIC_STALL_SEC = 8.0
    _MIC_RESTART_COOLDOWN_SEC = 20.0

    async def stream(self) -> AsyncIterator[str]:
        import time
        last_restart = 0.0
        while True:
            # 사람 없으면 잠깐 쉬고 재체크 (STT 호출 자체 차단)
            if not self._user_present():
                await asyncio.sleep(2.0)
                continue
            # 마이크 stall 감지 — 장시간 가동 시 USB/ALSA 스트림이 깨져 callback이
            # 멈추면 record_utterance가 영원히 None만 반환(먹통). callback 무응답이
            # 길면 스트림 재오픈으로 복구.
            age = getattr(self.mic, "frames_age_sec", lambda: 0.0)()
            now_mono = time.monotonic()
            if (age > self._MIC_STALL_SEC
                    and now_mono - last_restart > self._MIC_RESTART_COOLDOWN_SEC):
                log.warning(
                    f"마이크 stall 감지 (callback {age:.0f}s 무응답) — 스트림 재시작"
                )
                if hasattr(self.mic, "restart"):
                    self.mic.restart()
                last_restart = now_mono
                await asyncio.sleep(0.5)
                continue
            try:
                wav = await self.recorder.record_utterance(max_sec=self.max_sec)
            except Exception as e:
                log.warning(f"VAD record 실패: {e}")
                await asyncio.sleep(0.5)
                continue
            if not wav:
                # 발화 없음 — 잠깐 양보 후 다음 라운드
                await asyncio.sleep(0.05)
                continue
            # 신호 너무 약하면 (마이크 noise/짧은 클릭) Whisper에 안 보냄 —
            # base level hallucination("고맙습니다" 등) 차단 + CPU/비용 절약.
            # peak < 400은 일반 마이크 floor 수준 — 실제 발화는 보통 1000+.
            wav_peak = _wav_peak(wav)
            if wav_peak < 400:
                log.debug(f"utterance too quiet — skip (peak={wav_peak})")
                continue
            # 마이크 게인이 낮으면 (저감도 USB 마이크) Whisper가 텍스트 못 뽑음.
            # WAV peak를 ~30000(clip 직전)으로 정규화.
            wav = _normalize_wav_peak(wav)
            try:
                text = await self.stt.transcribe(wav)
            except Exception as e:
                log.warning(f"STT 호출 실패: {e}")
                continue
            text = (text or "").strip()
            # 너무 짧으면 무시 (잡음/단발 음절)
            if len(text) <= 2:
                continue
            # 백엔드 무관 hallucination 필터 — OpenAI/local 둘 다 한국어 약한
            # 신호에 "감사합니다"/"구독 좋아요" 흔히 끼움
            if _is_hallucination(text):
                log.info(f'hallucination drop: "{text}"')
                continue
            yield text


# 콜백 시그니처: 발화 텍스트 받아서 처리
TranscriptHandler = Callable[[str], Coroutine[Any, Any, None]]


class AmbientListener:
    """STT 결과를 받아 등록된 핸들러들에게 전달.

    stt 인자가 None이면 명시적 비활성 (mock fallback 안 함). main_robot이
    OPENAI_API_KEY 보고 WhisperVADStreamer 주입.
    """

    def __init__(
        self,
        stt: Any | None = None,
        perception: Any | None = None,
    ) -> None:
        if stt is None:
            raise ValueError(
                "stt 인자 필수 — MockSTT() 또는 WhisperVADStreamer() 명시"
            )
        self.stt = stt
        # perception 주입 시 STT 결과마다 last_user_speech_at 갱신 →
        # agent가 변화 감지해 즉시 tick 트리거.
        self.perception = perception
        self.handlers: list[TranscriptHandler] = []
        # 시스템 명령 handler — 일반 transcript handler보다 먼저 호출.
        # 반환값이 truthy("consumed")면 그 발화는 conversation_log/perception에
        # 안 들어가고 일반 handler도 skip. agent가 "왜 사용자가 그런 말 했지?"
        # 라고 반응하는 거 차단 (예: "디버그 모드"는 wifi 전환 명령일 뿐).
        self.system_handlers: list[TranscriptHandler] = []

    def add_handler(self, handler: TranscriptHandler) -> None:
        self.handlers.append(handler)

    def add_system_handler(self, handler: TranscriptHandler) -> None:
        """시스템 명령 handler. truthy 반환 시 발화 'consumed' — 일반 흐름 skip."""
        self.system_handlers.append(handler)

    async def run(self) -> None:
        import time
        async for text in self.stt.stream():
            # 1) 시스템 명령 먼저 — consumed면 일반 흐름 전부 skip
            consumed = False
            for h in self.system_handlers:
                try:
                    result = await h(text)
                    if result:
                        consumed = True
                        break
                except Exception as e:
                    log.warning(f"system handler 에러: {e}")
            if consumed:
                log.info(f'system command consumed: "{text}" — log/agent skip')
                continue
            # 2) 일반 사용자 발화 — conversation_log 기록 + perception 갱신 + handlers
            try:
                memory.log_user(text, kind="ambient")
            except Exception as e:
                log.debug(f"conversation log 실패: {e}")
            now_ts = time.time()
            if self.perception is not None:
                self.perception.last_user_speech_at = now_ts
                # 호명 감지 — agent가 cooldown 무조건 bypass + 즉시 응답
                if _is_calling_robot(text):
                    self.perception.last_user_called_at = now_ts
                    log.info(f'📣 호명 감지: "{text}"')
            for h in self.handlers:
                try:
                    await h(text)
                except Exception as e:
                    log.warning(f"handler 에러: {e}")
