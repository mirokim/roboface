"""audio_monitor 박수/음악 감지 + 마이크 broker 테스트."""

from __future__ import annotations

import math
import queue
import struct
import time

import pytest

from src.audio.audio_monitor import AudioMonitor, _frame_rms
from src.audio.mic import BYTES_PER_SAMPLE, FRAME_SAMPLES


def _pcm_with_amplitude(amp: int) -> bytes:
    """일정 진폭의 톤 — 30ms / 16kHz / int16."""
    n = FRAME_SAMPLES
    samples = []
    for i in range(n):
        # 200Hz sin
        v = int(math.sin(2 * math.pi * 200 * i / 16000) * amp)
        v = max(-32768, min(32767, v))
        samples.append(v)
    return struct.pack("<" + "h" * n, *samples)


def test_frame_rms_zero_for_silence():
    pcm = bytes(FRAME_SAMPLES * BYTES_PER_SAMPLE)
    assert _frame_rms(pcm) == 0.0


def test_frame_rms_increases_with_amplitude():
    quiet = _pcm_with_amplitude(500)
    loud = _pcm_with_amplitude(10000)
    assert _frame_rms(loud) > _frame_rms(quiet) * 5


class _FakeMic:
    """add_subscriber/remove_subscriber만 지원하는 mic stub."""
    def __init__(self):
        self.queues: list[queue.Queue] = []

    def add_subscriber(self, maxsize: int = 200):
        q: queue.Queue = queue.Queue(maxsize=maxsize)
        self.queues.append(q)
        return q

    def remove_subscriber(self, q):
        if q in self.queues:
            self.queues.remove(q)


def _drive_detector(mon: AudioMonitor, frames: list[bytes],
                    start_time: float = 0.0) -> float:
    """매 프레임 직접 처리. 16 frame마다 music check. 다음 호출용 끝 시각 반환."""
    t = start_time
    for i, pcm in enumerate(frames):
        rms = _frame_rms(pcm)
        if mon._detect_onset(rms, t):
            mon._onsets.append(t)
            mon._maybe_clap(t)
        if i % 16 == 0:
            mon._check_music(t)
        t += 0.03
    return t


def test_clap_detected_on_single_spike():
    captured: list = []
    mon = AudioMonitor(
        _FakeMic(),
        on_clap=lambda: captured.append("clap"),
        clap_absolute_min=500.0,
        clap_ratio=3.0,
    )
    # baseline 안정 — 1초간 작은 노이즈
    frames = [_pcm_with_amplitude(150) for _ in range(40)]
    # 한 frame만 큰 spike
    frames.append(_pcm_with_amplitude(8000))
    # 다시 조용
    frames.extend(_pcm_with_amplitude(150) for _ in range(20))

    _drive_detector(mon, frames)
    assert captured == ["clap"]


def test_music_detected_with_regular_beat():
    started: list[float] = []
    mon = AudioMonitor(
        _FakeMic(),
        on_music_start=lambda bpm: started.append(bpm),
        clap_absolute_min=500.0,
        clap_ratio=3.0,
        music_min_onsets=6,
    )
    # baseline 안정
    quiet = [_pcm_with_amplitude(150) for _ in range(15)]
    _drive_detector(mon, quiet)

    # 120 BPM = 0.5초 간격 → 33fps × 0.5 = 16 frame마다 spike
    # 8개 onset 만들기
    pattern = []
    for _ in range(10):
        # 0.5초 = 16 frames, 마지막 frame이 spike
        pattern.extend(_pcm_with_amplitude(150) for _ in range(15))
        pattern.append(_pcm_with_amplitude(7000))
    _drive_detector(mon, pattern)

    assert len(started) == 1
    bpm = started[0]
    # 추정 BPM이 90~150 사이
    assert 90 <= bpm <= 150


def test_music_stops_after_silence():
    started: list = []
    stopped: list = []
    mon = AudioMonitor(
        _FakeMic(),
        on_music_start=lambda bpm: started.append(bpm),
        on_music_stop=lambda: stopped.append("stop"),
        clap_absolute_min=500.0,
        clap_ratio=3.0,
        music_min_onsets=6,
        music_silence_to_stop_sec=1.0,
    )
    quiet = [_pcm_with_amplitude(150) for _ in range(15)]
    t = _drive_detector(mon, quiet)

    # 음악 시작 패턴
    pattern = []
    for _ in range(10):
        pattern.extend(_pcm_with_amplitude(150) for _ in range(15))
        pattern.append(_pcm_with_amplitude(7000))
    t = _drive_detector(mon, pattern, start_time=t)
    assert len(started) == 1

    # 그 후 2초 정적 (silence_to_stop_sec=1.0 초과)
    silence = [_pcm_with_amplitude(150) for _ in range(70)]
    _drive_detector(mon, silence, start_time=t)
    assert stopped == ["stop"]


def test_static_silence_no_false_clap():
    captured = []
    mon = AudioMonitor(
        _FakeMic(),
        on_clap=lambda: captured.append("clap"),
        clap_absolute_min=500.0,
    )
    frames = [_pcm_with_amplitude(120) for _ in range(80)]
    _drive_detector(mon, frames)
    assert captured == []


def test_microphone_broker_fans_out():
    """Microphone.add_subscriber + _callback이 모든 큐에 데이터 fan-out."""
    from src.audio.mic import Microphone

    # Microphone._load_backends 우회 (sounddevice 없어도 broker 자체는 테스트 가능)
    import types
    mic = types.SimpleNamespace()
    mic._queue = queue.Queue(maxsize=200)
    mic._subscribers = [mic._queue]
    import threading
    mic._lock = threading.Lock()
    # _callback이 신규로 보는 변환 플래그 — 변환 없이 그대로 fan-out
    mic._needs_convert = False

    # 진짜 Microphone의 _callback를 unbound로 호출
    other = Microphone.add_subscriber.__get__(mic)(maxsize=20)
    Microphone._callback.__get__(mic)(b"\x00\x10" * FRAME_SAMPLES, FRAME_SAMPLES, None, None)

    assert mic._queue.qsize() == 1
    assert other.qsize() == 1
