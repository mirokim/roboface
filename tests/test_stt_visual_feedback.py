"""STT 시각 피드백 — VADRecorder 콜백 + WhisperVADStreamer echo."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ─── VADRecorder 콜백 ───

def test_vadrecorder_fires_speech_callbacks_in_order():
    """on_speech_start는 발화 감지 시점에, on_speech_end는 종료 시점에 한 번씩."""
    import asyncio
    import wave
    from src.audio import mic as mic_mod
    from src.audio.mic import FRAME_SAMPLES, BYTES_PER_SAMPLE

    # webrtcvad 모킹: 처음 5프레임 무음, 그 다음 10프레임 speech, 다음 15프레임 무음
    class FakeVad:
        def __init__(self, *a, **k): pass
        def is_speech(self, frame, rate):
            FakeVad._count = getattr(FakeVad, "_count", 0) + 1
            n = FakeVad._count
            return 5 < n <= 15

    fake_module = MagicMock()
    fake_module.Vad = FakeVad
    with patch.object(mic_mod, "_load_backends", return_value=(MagicMock(), fake_module)):
        # 마이크 stub — frame() 호출 시 dummy PCM 반환
        fake_mic = MagicMock()
        fake_mic.sample_rate = 16000
        dummy_frame = b"\x00\x01" * FRAME_SAMPLES  # FRAME_SAMPLES*BYTES_PER_SAMPLE 바이트

        async def fake_frame(timeout=1.0):
            return dummy_frame
        fake_mic.frame = fake_frame

        events: list[str] = []
        rec = mic_mod.VADRecorder(
            fake_mic,
            silence_ms=300,   # 10 frames @ 30ms
            on_speech_start=lambda: events.append("start"),
            on_speech_end=lambda: events.append("end"),
        )

        FakeVad._count = 0
        wav = asyncio.run(rec.record_utterance(max_sec=2.0))

    assert wav is not None and len(wav) > 0
    assert events == ["start", "end"], f"콜백 순서: {events}"


def test_vadrecorder_fires_end_on_exception():
    """예외/취소로 record 종료해도 start 후엔 end가 무조건 발동 — indicator stuck on 방지."""
    import asyncio
    from src.audio import mic as mic_mod
    from src.audio.mic import FRAME_SAMPLES

    class FakeVad:
        _count = 0
        def __init__(self, *a, **k): pass
        def is_speech(self, frame, rate):
            FakeVad._count += 1
            return FakeVad._count > 2

    fake_module = MagicMock()
    fake_module.Vad = FakeVad
    with patch.object(mic_mod, "_load_backends", return_value=(MagicMock(), fake_module)):
        fake_mic = MagicMock()
        fake_mic.sample_rate = 16000
        call_count = {"n": 0}

        async def boom_after_start(timeout=1.0):
            call_count["n"] += 1
            if call_count["n"] > 5:
                raise RuntimeError("simulated mic failure")
            return b"\x00\x01" * FRAME_SAMPLES

        fake_mic.frame = boom_after_start

        events: list[str] = []
        rec = mic_mod.VADRecorder(
            fake_mic,
            on_speech_start=lambda: events.append("start"),
            on_speech_end=lambda: events.append("end"),
        )

        FakeVad._count = 0
        with pytest.raises(RuntimeError):
            asyncio.run(rec.record_utterance(max_sec=2.0))

    # start 한 번, end도 한 번 (finally가 발동)
    assert events == ["start", "end"]


def test_vadrecorder_no_end_if_never_started():
    """발화 감지 자체가 안 됐으면 end도 호출 X."""
    import asyncio
    from src.audio import mic as mic_mod

    class FakeVad:
        def __init__(self, *a, **k): pass
        def is_speech(self, frame, rate):
            return False  # 영원히 무음

    fake_module = MagicMock()
    fake_module.Vad = FakeVad
    with patch.object(mic_mod, "_load_backends", return_value=(MagicMock(), fake_module)):
        fake_mic = MagicMock()
        fake_mic.sample_rate = 16000

        async def fake_frame(timeout=1.0):
            return b"\x00\x01" * mic_mod.FRAME_SAMPLES
        fake_mic.frame = fake_frame

        events: list[str] = []
        rec = mic_mod.VADRecorder(
            fake_mic,
            start_timeout_sec=0.1,
            on_speech_start=lambda: events.append("start"),
            on_speech_end=lambda: events.append("end"),
        )

        result = asyncio.run(rec.record_utterance(max_sec=1.0))

    assert result is None
    assert events == []   # 발화 없으면 둘 다 X


def test_vadrecorder_callback_exception_swallowed():
    """콜백이 raise해도 녹음 자체엔 영향 X."""
    import asyncio
    from src.audio import mic as mic_mod

    class FakeVad:
        _count = 0
        def __init__(self, *a, **k): pass
        def is_speech(self, frame, rate):
            FakeVad._count += 1
            return 2 < FakeVad._count <= 6

    fake_module = MagicMock()
    fake_module.Vad = FakeVad
    with patch.object(mic_mod, "_load_backends", return_value=(MagicMock(), fake_module)):
        fake_mic = MagicMock()
        fake_mic.sample_rate = 16000

        async def fake_frame(timeout=1.0):
            return b"\x00\x01" * mic_mod.FRAME_SAMPLES
        fake_mic.frame = fake_frame

        def boom():
            raise RuntimeError("face crashed")

        rec = mic_mod.VADRecorder(
            fake_mic,
            silence_ms=120,
            on_speech_start=boom,
            on_speech_end=boom,
        )
        FakeVad._count = 0
        wav = asyncio.run(rec.record_utterance(max_sec=2.0))
    # 콜백이 깨져도 wav 정상 반환
    assert wav is not None


# ─── WhisperVADStreamer 시각 피드백 ───

class _FaceStub:
    def __init__(self) -> None:
        self.recording = False
        self.speech_text: str | None = None
        self.speech_duration: float = 0.0

    def show_speech(self, text: str, duration_sec: float) -> None:
        self.speech_text = text
        self.speech_duration = duration_sec


def test_streamer_callbacks_toggle_face_recording():
    """on_speech_start/end 콜백이 face.recording을 토글."""
    from src.tasks.ambient_listener import WhisperVADStreamer

    face = _FaceStub()
    fake_mic = MagicMock()
    fake_mic.sample_rate = 16000
    with patch("src.audio.mic.VADRecorder"):
        s = WhisperVADStreamer(fake_mic, MagicMock(), face=face)

    assert face.recording is False
    s._on_speech_start()
    assert face.recording is True
    s._on_speech_end()
    assert face.recording is False


def test_streamer_echo_writes_speech_bubble():
    """STT 결과가 face.show_speech로 echo."""
    from src.tasks.ambient_listener import WhisperVADStreamer

    face = _FaceStub()
    fake_mic = MagicMock()
    fake_mic.sample_rate = 16000
    with patch("src.audio.mic.VADRecorder"):
        s = WhisperVADStreamer(fake_mic, MagicMock(), face=face)
    s._echo_transcript("내일 회의 있어")

    assert face.speech_text is not None
    assert "내일 회의 있어" in face.speech_text
    # prefix가 붙어 다른 발화와 구분됨
    assert face.speech_text.startswith(WhisperVADStreamer.ECHO_PREFIX)
    assert face.speech_duration == pytest.approx(WhisperVADStreamer.ECHO_DURATION_SEC)


def test_streamer_no_face_no_crash():
    """face 미주입 시에도 콜백/echo 모두 안전."""
    from src.tasks.ambient_listener import WhisperVADStreamer

    fake_mic = MagicMock()
    fake_mic.sample_rate = 16000
    with patch("src.audio.mic.VADRecorder"):
        s = WhisperVADStreamer(fake_mic, MagicMock(), face=None)
    # 다 no-op이어야 함
    s._on_speech_start()
    s._on_speech_end()
    s._echo_transcript("test")
