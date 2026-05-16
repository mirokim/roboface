"""TTS — OpenAI TTS + 립싱크용 RMS 추출 + 재생.

OpenAI TTS는 MP3로 응답하므로,
1) MP3 → PCM 디코딩 (pydub + ffmpeg, 또는 audioread fallback)
2) RMS 윈도우(~50ms)를 계산해 입 모양 진폭으로 변환
3) sounddevice로 재생 + face.mouth_state.talk_amplitude 갱신

graceful fallback:
- openai 미설치 → TTSError
- pydub/ffmpeg 미설치 → 재생 불가, 텍스트만 로그 + fake mouth animation
"""

from __future__ import annotations

import asyncio
import os
import random

from src.face.expressions import MouthShape
from src.face.renderer import FaceState
from src.utils.logger import get_logger

log = get_logger("tts")


class TTSError(RuntimeError):
    pass


def _shape_for_amp(amp: float) -> MouthShape:
    # MouthShape에 CLOSED 가 없으므로 NEUTRAL (가벼운 호)를 닫힌 입으로 사용.
    if amp < 0.15:
        return MouthShape.NEUTRAL
    if amp < 0.35:
        return MouthShape.OPEN_SMALL
    if amp < 0.65:
        return MouthShape.OPEN_MID
    return MouthShape.OPEN_LARGE


class OpenAITTS:
    """OpenAI TTS.

    model: "gpt-4o-mini-tts" (저렴/빠름) or "tts-1" (구버전)
    voice: "alloy" / "echo" / "fable" / "onyx" / "nova" / "shimmer" / "coral" / "sage"
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini-tts",
        voice: str = "nova",
        format: str = "mp3",
    ) -> None:
        api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise TTSError("OPENAI_API_KEY 환경변수 필요")
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as e:
            raise TTSError(f"openai 미설치: {e}") from e
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.voice = voice
        self.format = format

    async def synthesize(self, text: str) -> bytes:
        """텍스트 → MP3 바이트."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> bytes:
        try:
            with self._client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=self.voice,
                input=text,
                response_format=self.format,
            ) as resp:
                return resp.read()
        except Exception as e:
            log.warning(f"TTS 합성 실패: {e}")
            return b""


def _decode_mp3_to_pcm(mp3: bytes) -> tuple[bytes, int]:
    """MP3 → 16-bit mono PCM @ 24kHz.

    pydub + ffmpeg 필요. 실패 시 (b'', 0).
    """
    try:
        from pydub import AudioSegment  # type: ignore[import-not-found]
    except ImportError:
        log.warning("pydub 없음 — TTS 재생 불가, mouth animation만")
        return b"", 0
    import io
    try:
        seg = AudioSegment.from_file(io.BytesIO(mp3), format="mp3")
        seg = seg.set_channels(1).set_frame_rate(24000).set_sample_width(2)
        return seg.raw_data, seg.frame_rate
    except Exception as e:
        log.warning(f"MP3 디코딩 실패 (ffmpeg 설치 확인): {e}")
        return b"", 0


def _compute_rms_envelope(pcm: bytes, sample_rate: int,
                          window_ms: int = 50) -> list[tuple[float, float]]:
    """PCM int16 → [(time_sec, normalized_amp 0..1), ...].

    윈도우별 RMS를 16384 기준으로 정규화.
    """
    import array
    samples = array.array("h")
    samples.frombytes(pcm)
    if not samples:
        return []
    win = max(1, sample_rate * window_ms // 1000)
    envelope: list[tuple[float, float]] = []
    peak = 1.0
    for i in range(0, len(samples), win):
        chunk = samples[i:i + win]
        if not chunk:
            break
        rms = (sum(s * s for s in chunk) / len(chunk)) ** 0.5
        peak = max(peak, rms)
        envelope.append((i / sample_rate, rms))
    # normalize
    return [(t, min(1.0, r / peak)) for t, r in envelope]


async def speak(
    face: FaceState,
    text: str,
    tts: OpenAITTS | None = None,
) -> None:
    """TTS로 발화 + 립싱크 + 말풍선.

    tts=None이면 fake animation으로 폴백.
    """
    if not text:
        return
    log.info(f'speak: "{text}"')

    if tts is None:
        # 폴백: 텍스트 길이 비례 fake animation
        await _fake_speak(face, text)
        return

    mp3 = await tts.synthesize(text)
    if not mp3:
        await _fake_speak(face, text)
        return

    pcm, sr = _decode_mp3_to_pcm(mp3)
    if not pcm or sr == 0:
        # 디코딩 실패 — animation만
        await _fake_speak(face, text)
        return

    envelope = _compute_rms_envelope(pcm, sr, window_ms=50)
    duration = len(pcm) / 2 / sr  # int16 mono

    # 말풍선
    face.show_speech(text, duration)

    # 재생 시작 (비동기) + envelope 따라 mouth 갱신
    playback = asyncio.create_task(_play_pcm(pcm, sr))
    saved_shape = face.mouth_state.shape
    try:
        start = asyncio.get_event_loop().time()
        for t, amp in envelope:
            now = asyncio.get_event_loop().time() - start
            if t > now:
                await asyncio.sleep(t - now)
            face.mouth_state.shape = _shape_for_amp(amp)
            face.mouth_state.talk_amplitude = amp
        # 잔여 시간
        elapsed = asyncio.get_event_loop().time() - start
        if elapsed < duration:
            await asyncio.sleep(duration - elapsed)
    finally:
        face.mouth_state.shape = saved_shape
        face.mouth_state.talk_amplitude = 0.0
        # 말풍선은 show_speech가 duration+1.0초로 잡아둠 — renderer가 만료 시 자동 클리어
        try:
            await playback
        except Exception:
            pass


async def _play_pcm(pcm: bytes, sample_rate: int) -> None:
    """sounddevice로 PCM 재생 (블로킹 → executor)."""
    try:
        import numpy as np  # type: ignore[import-not-found]
        import sounddevice as sd  # type: ignore[import-not-found]
    except ImportError as e:
        log.warning(f"재생 백엔드 없음: {e}")
        return
    loop = asyncio.get_running_loop()
    arr = np.frombuffer(pcm, dtype=np.int16)
    await loop.run_in_executor(None, lambda: (sd.play(arr, sample_rate), sd.wait()))


async def _fake_speak(face: FaceState, text: str,
                      duration_per_char: float = 0.06) -> None:
    """TTS 사용 불가 시 길이 기반 더미 입 모양 + 말풍선."""
    duration = max(0.5, len(text) * duration_per_char)
    face.show_speech(text, duration)
    end = asyncio.get_event_loop().time() + duration
    saved_shape = face.mouth_state.shape
    while asyncio.get_event_loop().time() < end:
        face.mouth_state.shape = random.choice([
            MouthShape.OPEN_SMALL,
            MouthShape.OPEN_MID,
            MouthShape.OPEN_LARGE,
        ])
        face.mouth_state.talk_amplitude = random.uniform(0.2, 0.9)
        await asyncio.sleep(random.uniform(0.08, 0.14))
    face.mouth_state.shape = saved_shape
    face.mouth_state.talk_amplitude = 0.0
