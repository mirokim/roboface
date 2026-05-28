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
        self.sample_rate = sample_rate   # downstream(target) rate
        self._sd, _ = _load_backends()
        self._stream = None
        # 기본 큐 — frame() 호환성 유지
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=200)
        self._subscribers: list[queue.Queue[bytes]] = [self._queue]
        self._lock = threading.Lock()
        # 디바이스 native format 협상 — 일부 USB 마이크(예: COMS CM421)는
        # 16kHz mono 지원 X. native rate/channels로 캡처 후 numpy resample.
        self._native_rate = sample_rate
        self._native_channels = 1
        self._needs_convert = False
        self._negotiate_format()

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

    def _negotiate_format(self) -> None:
        """디바이스가 target rate/channels로 직접 못 열리면 native로 fallback.

        예: COMS CM421은 44.1kHz stereo만 — 16k mono 요청 시 PortAudio가
        PaInvalidSampleRate 던짐. native(44.1k 2ch)로 캡처 후 callback에서
        numpy로 mono 변환 + 다운샘플.
        """
        try:
            info = self._sd.query_devices(self.device, "input")
        except Exception as e:
            log.debug(f"디바이스 query 실패 — target 그대로: {e}")
            return
        max_ch = int(info.get("max_input_channels", 1))
        default_rate = int(info.get("default_samplerate", self.sample_rate))
        # 일단 target 그대로 가능한지 빠르게 try
        try:
            self._sd.check_input_settings(
                device=self.device, channels=1, samplerate=self.sample_rate,
                dtype="int16",
            )
            return   # native가 target 지원 — 그대로 가자
        except Exception:
            pass
        # 변환 필요 — native rate/channels로 캡처
        self._native_rate = default_rate
        self._native_channels = min(max_ch, 2) if max_ch > 0 else 1
        self._needs_convert = True
        log.info(
            f"마이크 native format으로 fallback: "
            f"{self._native_rate}Hz {self._native_channels}ch "
            f"→ {self.sample_rate}Hz 1ch (resample)"
        )

    def _convert_native_to_target(self, indata) -> bytes:
        """native rate/channels → target rate mono. numpy.interp 기반 단순 resample.

        음성 대역(80-8kHz)엔 충분. 고품질 anti-aliasing 필요하면 scipy 추가.
        """
        import numpy as np
        arr = np.frombuffer(
            bytes(indata), dtype=np.int16,
        ).reshape(-1, self._native_channels)
        # stereo → mono (평균)
        if self._native_channels > 1:
            mono = arr.mean(axis=1).astype(np.float32)
        else:
            mono = arr[:, 0].astype(np.float32)
        # rate convert (linear interp)
        if self._native_rate != self.sample_rate:
            n_out = int(len(mono) * self.sample_rate / self._native_rate)
            if n_out <= 1:
                return b""
            x_in = np.linspace(0, 1, len(mono), endpoint=False)
            x_out = np.linspace(0, 1, n_out, endpoint=False)
            mono = np.interp(x_out, x_in, mono)
        return mono.astype(np.int16).tobytes()

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ARG002
        if status:
            log.debug(f"sd status: {status}")
        # indata: int16 numpy array shape (frames, native_channels)
        if self._needs_convert:
            data = self._convert_native_to_target(indata)
        else:
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
        # native rate가 다르면 그 rate로 캡처 (callback에서 변환).
        # blocksize도 native rate 기준 30ms 프레임으로.
        open_rate = self._native_rate if self._needs_convert else self.sample_rate
        open_channels = (
            self._native_channels if self._needs_convert else 1
        )
        open_blocksize = (open_rate * FRAME_MS) // 1000
        try:
            self._stream = self._sd.RawInputStream(
                samplerate=open_rate,
                blocksize=open_blocksize,
                device=self.device,
                channels=open_channels,
                dtype="int16",
                callback=self._callback,
            )
            self._stream.start()
        except Exception as e:
            # sounddevice.PortAudioError 등 — 디바이스 없음/잘못됨
            raise MicCaptureError(
                f"오디오 디바이스 열기 실패 (device={self.device}): {e}"
            ) from e
        log.info(
            f"마이크 시작 (device={self.device}, "
            f"{open_rate}Hz {open_channels}ch"
            + (f" → {self.sample_rate}Hz 1ch" if self._needs_convert else "")
            + ")"
        )
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
