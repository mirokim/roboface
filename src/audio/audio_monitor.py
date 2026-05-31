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


def _frame_rms_and_peak(pcm: bytes) -> tuple[float, float]:
    """30ms int16 PCM → (RMS, peak abs). 둘 다 0~32768.

    박수처럼 짧은 transient(~10ms)는 30ms RMS 평균에 희석돼 약하게 잡힘 —
    peak abs를 따로 봐서 transient에 robust. RMS는 baseline(EMA) 추적용.
    """
    if len(pcm) < 2:
        return 0.0, 0.0
    n = len(pcm) // 2
    acc = 0
    peak = 0
    for i in range(0, len(pcm), 2):
        lo = pcm[i]
        hi = pcm[i + 1]
        v = (hi << 8) | lo
        if v >= 0x8000:
            v -= 0x10000
        acc += v * v
        av = -v if v < 0 else v
        if av > peak:
            peak = av
    return math.sqrt(acc / n), float(peak)


def _frame_rms(pcm: bytes) -> float:
    """하위 호환 — 기존 호출자/테스트 위해 RMS만 반환."""
    return _frame_rms_and_peak(pcm)[0]


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
        # 약한 마이크 + transient 박수도 잡히게 완화 (RMS + peak 두 갈래).
        # RMS는 baseline 대비 ratio 비교 — 환경 노이즈 적응형
        # peak는 절대 임계 — transient(짧고 강한) 신호 보호 (박수는 ~10ms peak)
        # 둘 중 하나라도 만족하면 onset.
        # CM421 같은 저게인 마이크 실측: 평소 abs_peak 600~1300, 박수 시도 2300~2500.
        # peak_min 1500이면 박수 확실히 잡고, 일상 소음(키보드/식기) 일부 false +
        # 가능. false 증가하면 alsamixer로 게인 올리고 임계 다시 ↑.
        clap_ratio: float = 3.0,
        clap_absolute_min: float = 350.0,    # RMS 임계 (낮춤 — 약한 박수 cover)
        clap_peak_min: float = 1500.0,       # peak 임계 — transient 단독 트리거
        # 0.4 → 0.25 — 두 번 빠르게 친 박수(보통 200~400ms 간격)도 두 번 다
        # 발동하도록. _refractory_until(80ms)이 echo 방지 따로 처리.
        clap_cooldown_sec: float = 0.25,
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
        self.clap_peak_min = clap_peak_min
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

    def _detect_onset(self, rms: float, now: float, peak: float = 0.0) -> bool:
        """현재 frame이 onset인지 판단 + baseline 갱신.

        두 갈래 onset 조건 (둘 중 하나라도):
          (A) RMS ratio: rms ≥ clap_absolute_min AND rms / baseline ≥ clap_ratio
              → 환경 적응형 — 평소 조용한 방에서 살짝만 spike도 잡음
          (B) Peak transient: peak ≥ clap_peak_min
              → 박수처럼 짧고 강한 transient는 30ms RMS에 희석돼도 peak는 큼
        """
        ratio = rms / max(1.0, self._baseline)
        rms_onset = (
            rms >= self.clap_absolute_min
            and ratio >= self.clap_ratio
        )
        peak_onset = peak >= self.clap_peak_min
        is_onset = (
            (rms_onset or peak_onset)
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
            log.debug("onset 발생했지만 music_playing — 박수 분류 skip")
            return
        if now - self._last_clap_at < self.clap_cooldown_sec:
            log.debug(
                f"onset 발생했지만 cooldown 안 — 박수 skip "
                f"({now - self._last_clap_at:.2f}s < {self.clap_cooldown_sec})"
            )
            return
        recent = [t for t in self._onsets if now - t < 1.0]
        if len(recent) <= 2:
            # 박수 — 단발성 큰 소리
            self._last_clap_at = now
            log.info(f"👏 박수 감지 (직전 1초 onset {len(recent)}개, baseline≈{self._baseline:.0f})")
            if self.on_clap:
                try:
                    self.on_clap()
                except Exception as e:
                    log.warning(f"on_clap 핸들러 에러: {e}")
        else:
            log.debug(f"onset {len(recent)}개 — 박수 아닌 패턴 (음악 후보)")

    async def run(self) -> None:
        """blocking loop — mic에서 frame을 받아 처리."""
        log.info(
            f"audio_monitor 시작 (rms_min={self.clap_absolute_min}, "
            f"rms_ratio={self.clap_ratio}, peak_min={self.clap_peak_min}, "
            f"cooldown={self.clap_cooldown_sec}s)"
        )
        q = self.mic.add_subscriber()
        frame_count = 0
        rms_peak_window = 0.0    # 최근 5초 max RMS — baseline 위 수준 보기
        abs_peak_window = 0.0    # 최근 5초 max abs(sample) — transient 감지 진단
        try:
            while not self._stop.is_set():
                frame = await pop_frame(q, timeout=1.0)
                if frame is None or len(frame) != FRAME_SAMPLES * BYTES_PER_SAMPLE:
                    continue
                rms, peak = _frame_rms_and_peak(frame)
                now = time.monotonic()
                rms_peak_window = max(rms_peak_window, rms)
                abs_peak_window = max(abs_peak_window, peak)
                if self._detect_onset(rms, now, peak=peak):
                    # 모든 onset 로그 — RMS/peak 어느 갈래로 잡혔는지
                    log.info(
                        f"onset: rms={rms:.0f} peak={peak:.0f} "
                        f"baseline={self._baseline:.0f} "
                        f"ratio={rms / max(1.0, self._baseline):.1f}"
                    )
                    self._onsets.append(now)
                    self._maybe_clap(now)
                frame_count += 1
                # 매 ~0.5초 (16 frame × 30ms) music 상태 평가 (silence 동안에도 호출)
                if frame_count % 16 == 0:
                    self._check_music(now)
                # 매 ~5초 baseline + peak 로그 — 평소 마이크 신호 수준 확인.
                # 박수가 안 잡히는 진단:
                #   abs_peak < clap_peak_min(3000) 이면 마이크 신호 약해 transient
                #   감지도 어려움. 임계 더 낮추거나 OS 게인 올려야.
                if frame_count % 165 == 0:
                    log.info(
                        f"audio_monitor 5s 통계: baseline≈{self._baseline:.0f}, "
                        f"rms_peak={rms_peak_window:.0f}/{self.clap_absolute_min:.0f}, "
                        f"abs_peak={abs_peak_window:.0f}/{self.clap_peak_min:.0f}"
                    )
                    rms_peak_window = 0.0
                    abs_peak_window = 0.0
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
