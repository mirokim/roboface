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
    """

    def __init__(
        self,
        mic: Any,
        stt: Any,
        perception: Any | None = None,
        max_sec: float = 15.0,
        silence_ms: int = 700,
        aggressiveness: int = 2,
    ) -> None:
        from src.audio.mic import VADRecorder
        self.mic = mic
        self.stt = stt
        self.perception = perception
        self.recorder = VADRecorder(
            mic, aggressiveness=aggressiveness,
            silence_ms=silence_ms,
            start_timeout_sec=2.0,  # 짧게 — 사람 없으면 빨리 빠져나와 가드 재체크
        )
        self.max_sec = max_sec

    def _user_present(self) -> bool:
        if self.perception is None:
            return True
        return bool(getattr(self.perception, "user_present", True))

    async def stream(self) -> AsyncIterator[str]:
        while True:
            # 사람 없으면 잠깐 쉬고 재체크 (STT 호출 자체 차단)
            if not self._user_present():
                await asyncio.sleep(2.0)
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

    def add_handler(self, handler: TranscriptHandler) -> None:
        self.handlers.append(handler)

    async def run(self) -> None:
        import time
        async for text in self.stt.stream():
            try:
                memory.log_user(text, kind="ambient")
            except Exception as e:
                log.debug(f"conversation log 실패: {e}")
            # agent가 즉시 인지하도록 perception 갱신
            if self.perception is not None:
                self.perception.last_user_speech_at = time.time()
            for h in self.handlers:
                try:
                    await h(text)
                except Exception as e:
                    log.warning(f"handler 에러: {e}")
