"""음성 모듈 단위 테스트 — 하드웨어/네트워크 없이 import + 폴백 검증."""

from __future__ import annotations

import asyncio
import struct

import pytest

from src.audio.tts import (
    _compute_rms_envelope,
    _fake_speak,
)
from src.face.expressions import MouthShape, NEUTRAL
from src.face.mouth import shape_for_amp
from src.face.renderer import FaceState


def test_shape_for_amp_thresholds():
    # BehaviorConfig.mouth_amp_thresholds = (0.15, 0.35, 0.65)
    assert shape_for_amp(0.0) == MouthShape.NEUTRAL
    assert shape_for_amp(0.1) == MouthShape.NEUTRAL
    assert shape_for_amp(0.2) == MouthShape.OPEN_SMALL
    assert shape_for_amp(0.5) == MouthShape.OPEN_MID
    assert shape_for_amp(0.9) == MouthShape.OPEN_LARGE


def test_rms_envelope_silence():
    pcm = b"\x00\x00" * 16000  # 1초 무음
    env = _compute_rms_envelope(pcm, 16000, window_ms=50)
    assert len(env) == 20
    # 모두 0 RMS이지만 peak=1.0으로 정규화돼 0.0이어야 함
    for _, amp in env:
        assert amp == 0.0


def test_rms_envelope_normalizes_to_peak():
    # 한 윈도우만 큰 값
    samples = [0] * 8000 + [10000] * 800 + [0] * 7200
    pcm = struct.pack("<" + "h" * len(samples), *samples)
    env = _compute_rms_envelope(pcm, 16000, window_ms=50)
    peak = max(a for _, a in env)
    assert peak == pytest.approx(1.0)


def test_fake_speak_restores_mouth():
    face = FaceState(expression=NEUTRAL)
    saved = face.mouth_state.shape
    asyncio.run(_fake_speak(face, "안녕", duration_per_char=0.01))
    assert face.mouth_state.shape == saved
    assert face.mouth_state.talk_amplitude == 0.0


def test_voice_assistant_import():
    """voice_assistant가 hardware 없이도 import 되어야 main_robot이 깨지지 않음."""
    from src.tasks.voice_assistant import VoiceAssistant, run_voice_assistant
    assert VoiceAssistant is not None
    assert run_voice_assistant is not None


def test_wake_word_module_import():
    from src.audio.wake_word import PorcupineWakeWord, WakeWordError
    assert PorcupineWakeWord is not None
    assert issubclass(WakeWordError, RuntimeError)


def test_stt_module_import():
    from src.audio.stt import OpenAIWhisperSTT, STTError
    assert OpenAIWhisperSTT is not None
    assert issubclass(STTError, RuntimeError)


def test_mic_module_import():
    from src.audio.mic import Microphone, VADRecorder, MicCaptureError
    assert Microphone is not None
    assert VADRecorder is not None
    assert issubclass(MicCaptureError, RuntimeError)


def test_voice_assistant_graceful_when_no_mic(monkeypatch):
    """마이크 라이브러리 없을 때 task가 깔끔히 종료되어야 함."""
    from src.audio import mic as mic_mod

    def fake_load():
        raise mic_mod.MicCaptureError("no sounddevice for test")

    monkeypatch.setattr(mic_mod, "_load_backends", fake_load)

    from src.brain.state_machine import StateContext
    from src.tasks.voice_assistant import VoiceAssistant

    ctx = StateContext()
    face = FaceState(expression=NEUTRAL)
    va = VoiceAssistant(ctx, face)

    # run()이 예외 없이 종료되어야 함
    asyncio.run(va.run())
