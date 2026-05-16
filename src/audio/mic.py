"""마이크 캡처 + VAD (Voice Activity Detection).

USB 스피커폰의 마이크 입력을 16kHz/mono로 받아,
webrtcvad로 발화 구간만 잘라낸다.

graceful fallback: sounddevice/webrtcvad 미설치 시 NotImplementedError.
"""

from __future__ import annotations

import asyncio
import collections
import queue
import threading
from typing import Any

from src.utils.logger import get_logger

log = get_logger("mic")

# 표준 음성 처리 파라미터
SAMPLE_RATE = 16000           # Whisper/VAD 모두 16k 권장
FRAME_MS = 30                 # webrtcvad는 10/20/30ms만 허용
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480 samples
BYTES_PER_SAMPLE = 2          # int16


class MicCaptureError(RuntimeError):
    """마이크 캡처 자체가 불가능 (라이브러리 없음 / 디바이스 없음)."""


def _load_backends() -> tuple[Any, Any]:
    try:
        import sounddevice as sd  # type: ignore[import-not-found]
        import webrtcvad  # type: ignore[import-not-found]
    except ImportError as e:
        raise MicCaptureError(
            f"sounddevice/webrtcvad 미설치: {e}. "
            "pip install sounddevice webrtcvad"
        ) from e
    return sd, webrtcvad


class Microphone:
    """비동기 마이크 — 30ms 프레임 단위 PCM int16 바이트 yield.

    Usage:
        mic = Microphone(device=None)
        with mic.open():
            async for frame in mic.frames():
                ...
    """

    def __init__(self, device: int | str | None = None,
                 sample_rate: int = SAMPLE_RATE) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self._sd, _ = _load_backends()
        self._stream = None
        # 기본 큐 — frame() 호환성 유지
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=200)
        self._subscribers: list[queue.Queue[bytes]] = [self._queue]
        self._lock = threading.Lock()

    def add_subscriber(self, maxsize: int = 200) -> queue.Queue[bytes]:
        """별도 소비자가 자체 큐를 받음. (voice_assistant + audio_monitor 동시)."""
        q: queue.Queue[bytes] = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subscribers.append(q)
        return q

    def remove_subscriber(self, q: queue.Queue[bytes]) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ARG002
        if status:
            log.debug(f"sd status: {status}")
        # indata is int16 numpy array shape (frames, 1)
        data = bytes(indata)
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(data)
            except queue.Full:
                # 드롭 — 최신 유지
                try:
                    q.get_nowait()
                    q.put_nowait(data)
                except queue.Empty:
                    pass

    def open(self):  # context manager
        return self

    def __enter__(self) -> "Microphone":
        self._stream = self._sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=FRAME_SAMPLES,
            device=self.device,
            channels=1,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()
        log.info(f"마이크 시작 (device={self.device}, {self.sample_rate}Hz)")
        return self

    def __exit__(self, *exc) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        log.info("마이크 정지")

    async def frame(self, timeout: float = 1.0) -> bytes | None:
        """30ms PCM 프레임 한 개 (기본 큐). 타임아웃이면 None."""
        return await pop_frame(self._queue, timeout)


async def pop_frame(q: "queue.Queue[bytes]", timeout: float = 1.0) -> bytes | None:
    """임의의 subscriber 큐에서 frame 한 개 pop."""
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, lambda: q.get(timeout=timeout))
    except queue.Empty:
        return None


class VADRecorder:
    """webrtcvad로 발화 구간만 녹음 → WAV 바이트.

    사용 흐름:
        rec = VADRecorder(mic, aggressiveness=2)
        wav_bytes = await rec.record_utterance(max_sec=10)

    종료 조건:
    - silence_ms 만큼 무음이 연속이면 발화 끝으로 간주
    - max_sec 초과시 강제 종료
    - 30프레임(0.9s) 안에 발화 시작 안 하면 None
    """

    def __init__(
        self,
        mic: Microphone,
        aggressiveness: int = 2,
        silence_ms: int = 700,
        start_timeout_sec: float = 5.0,
    ) -> None:
        _, webrtcvad = _load_backends()
        self.mic = mic
        self.vad = webrtcvad.Vad(aggressiveness)
        self.silence_frames = silence_ms // FRAME_MS
        self.start_timeout_frames = int(start_timeout_sec * 1000 // FRAME_MS)

    async def record_utterance(self, max_sec: float = 10.0) -> bytes | None:
        max_frames = int(max_sec * 1000 // FRAME_MS)
        ring: collections.deque[bytes] = collections.deque(maxlen=10)  # pre-buffer
        recorded: list[bytes] = []
        started = False
        silent_count = 0
        idle_count = 0

        for _ in range(max_frames):
            frame = await self.mic.frame(timeout=1.0)
            if frame is None or len(frame) != FRAME_SAMPLES * BYTES_PER_SAMPLE:
                idle_count += 1
                if not started and idle_count > self.start_timeout_frames:
                    return None
                continue

            is_speech = self.vad.is_speech(frame, self.mic.sample_rate)

            if not started:
                ring.append(frame)
                if is_speech:
                    started = True
                    recorded.extend(ring)
                    log.debug("발화 시작 감지")
                else:
                    idle_count += 1
                    if idle_count > self.start_timeout_frames:
                        return None
            else:
                recorded.append(frame)
                if is_speech:
                    silent_count = 0
                else:
                    silent_count += 1
                    if silent_count >= self.silence_frames:
                        log.debug("발화 끝 감지")
                        break

        if not started or not recorded:
            return None

        pcm = b"".join(recorded)
        return _pcm_to_wav(pcm, self.mic.sample_rate)


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """16-bit mono PCM → WAV 바이트."""
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(BYTES_PER_SAMPLE)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()
