"""Audio monitor — 박수 감지 + 음악 비트 감지.

Microphone broker에 subscribe해서 30ms PCM 프레임을 받음.
RMS 시계열로:
- 단발성 큰 spike + 직후 정적 → 박수
- 5초+ 동안 일정 간격 onset → 음악 (BPM 추정)

이벤트는 콜백으로 전달 (on_clap, on_music_start, on_music_stop).
voice_assistant와 같은 Microphone 인스턴스 공유.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from collections.abc import Callable

from src.audio.mic import (
    BYTES_PER_SAMPLE, FRAME_MS, FRAME_SAMPLES, Microphone, pop_frame,
)
from src.utils.logger import get_logger

log = get_logger("audio_monitor")

ClapHandler = Callable[[], None]
MusicHandler = Callable[[float], None]  # BPM
StopHandler = Callable[[], None]


def _frame_rms(pcm: bytes) -> float:
    """30ms int16 PCM → RMS (0~32768)."""
    if len(pcm) < 2:
        return 0.0
    # 빠른 수동 계산 (numpy 미설치 환경 대응)
    n = len(pcm) // 2
    acc = 0
    for i in range(0, len(pcm), 2):
        # little-endian int16
        lo = pcm[i]
        hi = pcm[i + 1]
        v = (hi << 8) | lo
        if v >= 0x8000:
            v -= 0x10000
        acc += v * v
    return math.sqrt(acc / n)


class AudioMonitor:
    """박수 + 음악 감지기. mic에서 frame을 받아 분석.

    파라미터 튜닝 가이드:
    - clap_ratio: baseline 대비 몇 배 spike면 onset (기본 4.0)
    - clap_absolute_min: 절대 RMS 임계 (조용한 방에서 false-positive 방지)
    - clap_cooldown_sec: 같은 박수가 여러 번 감지되지 않게
    - music_window_sec: 음악 판단 시 onset 모으는 윈도우 (5초)
    - music_min_onsets: 그 안에 onset 최소 개수
    """

    def __init__(
        self,
        mic: Microphone,
        *,
        on_clap: ClapHandler | None = None,
        on_music_start: MusicHandler | None = None,
        on_music_stop: StopHandler | None = None,
        clap_ratio: float = 4.0,
        clap_absolute_min: float = 800.0,
        clap_cooldown_sec: float = 0.4,
        music_window_sec: float = 5.0,
        music_min_onsets: int = 8,
        music_bpm_range: tuple[float, float] = (60.0, 160.0),
        music_silence_to_stop_sec: float = 4.0,
    ) -> None:
        self.mic = mic
        self.on_clap = on_clap
        self.on_music_start = on_music_start
        self.on_music_stop = on_music_stop

        self.clap_ratio = clap_ratio
        self.clap_absolute_min = clap_absolute_min
        self.clap_cooldown_sec = clap_cooldown_sec
        self.music_window_sec = music_window_sec
        self.music_min_onsets = music_min_onsets
        self.music_bpm_range = music_bpm_range
        self.music_silence_to_stop_sec = music_silence_to_stop_sec

        # baseline EMA (slow follower)
        self._baseline = 200.0
        self._baseline_alpha = 0.02   # 매 30ms → 약 1.5초 시간 상수

        # onset 시각 buffer (음악 판단용)
        self._onsets: deque[float] = deque(maxlen=200)
        self._last_clap_at = 0.0
        self._refractory_until = 0.0   # 이 시각까지는 새 onset 무시
        self._music_playing = False
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    def _detect_onset(self, rms: float, now: float) -> bool:
        """현재 frame이 onset인지 판단 + baseline 갱신."""
        ratio = rms / max(1.0, self._baseline)
        is_onset = (
            rms >= self.clap_absolute_min
            and ratio >= self.clap_ratio
            and now >= self._refractory_until
        )
        # baseline은 onset 아닐 때만 갱신 (spike가 baseline 끌어올리지 않게)
        if not is_onset:
            self._baseline += self._baseline_alpha * (rms - self._baseline)
        else:
            # onset 직후 80ms 동안 새 onset 안 잡음 (echo 방지)
            self._refractory_until = now + 0.08
        return is_onset

    def _check_music(self, now: float) -> None:
        """onset 패턴 분석 → 음악 시작/종료 판단."""
        # window 밖 onset 제거
        cutoff = now - self.music_window_sec
        while self._onsets and self._onsets[0] < cutoff:
            self._onsets.popleft()

        if not self._music_playing:
            if len(self._onsets) >= self.music_min_onsets:
                # BPM 추정 — onset 간격의 median
                arr = list(self._onsets)
                intervals = [arr[i + 1] - arr[i] for i in range(len(arr) - 1)]
                if not intervals:
                    return
                intervals.sort()
                median = intervals[len(intervals) // 2]
                if median <= 0:
                    return
                bpm = 60.0 / median
                # 2x/0.5x 보정 (octave error 정도)
                while bpm < self.music_bpm_range[0]:
                    bpm *= 2
                while bpm > self.music_bpm_range[1]:
                    bpm /= 2
                lo, hi = self.music_bpm_range
                if lo <= bpm <= hi:
                    self._music_playing = True
                    log.info(f"🎵 음악 감지 — 추정 BPM={bpm:.1f}")
                    if self.on_music_start:
                        try:
                            self.on_music_start(bpm)
                        except Exception as e:
                            log.warning(f"on_music_start 핸들러 에러: {e}")
        else:
            # 재생 중 — 마지막 onset 이후 너무 오래면 종료 판단
            if not self._onsets:
                return
            last = self._onsets[-1]
            if now - last >= self.music_silence_to_stop_sec:
                self._music_playing = False
                log.info("🎵 음악 종료")
                if self.on_music_stop:
                    try:
                        self.on_music_stop()
                    except Exception as e:
                        log.warning(f"on_music_stop 핸들러 에러: {e}")

    def _maybe_clap(self, now: float) -> None:
        """가장 최근 onset이 단발성인지 — 박수 판단.

        조건: 직전 1초에 onset 1~2개만 (음악이면 더 많이 옴).
        """
        if self._music_playing:
            return
        if now - self._last_clap_at < self.clap_cooldown_sec:
            return
        recent = [t for t in self._onsets if now - t < 1.0]
        if len(recent) <= 2:
            # 박수 — 단발성 큰 소리
            self._last_clap_at = now
            log.info("👏 박수 감지")
            if self.on_clap:
                try:
                    self.on_clap()
                except Exception as e:
                    log.warning(f"on_clap 핸들러 에러: {e}")

    async def run(self) -> None:
        """blocking loop — mic에서 frame을 받아 처리."""
        log.info("audio_monitor 시작")
        q = self.mic.add_subscriber()
        frame_count = 0
        try:
            while not self._stop.is_set():
                frame = await pop_frame(q, timeout=1.0)
                if frame is None or len(frame) != FRAME_SAMPLES * BYTES_PER_SAMPLE:
                    continue
                rms = _frame_rms(frame)
                now = time.monotonic()
                if self._detect_onset(rms, now):
                    self._onsets.append(now)
                    self._maybe_clap(now)
                # 매 ~0.5초 (16 frame × 30ms) music 상태 평가 (silence 동안에도 호출)
                frame_count += 1
                if frame_count % 16 == 0:
                    self._check_music(now)
        finally:
            self.mic.remove_subscriber(q)
            log.info("audio_monitor 종료")


async def run_audio_monitor(
    mic: Microphone,
    *,
    on_clap: ClapHandler | None = None,
    on_music_start: MusicHandler | None = None,
    on_music_stop: StopHandler | None = None,
) -> None:
    """task 진입점."""
    monitor = AudioMonitor(
        mic,
        on_clap=on_clap,
        on_music_start=on_music_start,
        on_music_stop=on_music_stop,
    )
    await monitor.run()
