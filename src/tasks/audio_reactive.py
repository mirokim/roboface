"""audio_monitor 이벤트 → 표정/모션 반응.

- 박수: SURPRISED 0.4s → (사람 보이면 그쪽 시선) → 본래 표정
- 음악 시작: EXCITED + 자동 dance loop (감지 BPM)
- 음악 종료: dance 중단, 본래 표정 복귀
"""

from __future__ import annotations

import asyncio

from src.audio.audio_monitor import AudioMonitor
from src.audio.mic import Microphone
from src.brain.perception import PerceptionState
from src.brain.state_machine import State, StateContext, motion_busy_scope
from src.face import expressions as expr
from src.face.renderer import FaceState
from src.motion import poses
from src.motion.servos import ServoController
from src.tasks.reactive_face import flash_expression
from src.utils.logger import get_logger

log = get_logger("audio_reactive")


class AudioReactive:
    def __init__(
        self,
        mic: Microphone,
        face: FaceState,
        ctx: StateContext,
        perception: PerceptionState | None = None,
        servos: ServoController | None = None,
    ) -> None:
        self.mic = mic
        self.face = face
        self.ctx = ctx
        self.perception = perception
        self.servos = servos
        self._music_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # ─── 이벤트 핸들러 (audio_monitor 콜백) ───
    # 콜백은 마이크 백그라운드 스레드의 asyncio loop에서 호출됨 (audio_monitor가
    # asyncio.run loop 안에 있으므로 동일 loop). 그래서 직접 task 만들어도 OK.

    def _on_clap(self) -> None:
        # 대화 중엔 양보
        if self.ctx.state in (State.TALKING, State.LISTENING):
            return
        flash_expression(self.face, expr.SURPRISED, 0.5)
        # 살짝 끄덕 (놀란 척) — head_tracker와 충돌 안 하게 lock
        if self.servos is not None:
            try:
                asyncio.create_task(self._clap_nod())
            except RuntimeError:
                pass

    async def _clap_nod(self) -> None:
        async with motion_busy_scope(self.ctx):
            await poses.nod(self.servos, times=1)

    def _on_music_start(self, bpm: float) -> None:
        if self.ctx.state in (State.TALKING, State.LISTENING):
            return
        if self._music_task and not self._music_task.done():
            return
        log.info(f"🎵 음악 댄스 시작 BPM={bpm:.1f}")
        try:
            self._music_task = asyncio.create_task(
                self._music_dance_loop(bpm)
            )
        except RuntimeError:
            pass

    def _on_music_stop(self) -> None:
        if self._music_task and not self._music_task.done():
            self._music_task.cancel()
            self._music_task = None

    async def _music_dance_loop(self, bpm: float) -> None:
        """음악 끝날 때까지 dance 반복. cancel되면 정리."""
        # BPM 범위 클램프
        bpm = max(70.0, min(160.0, bpm))
        self.face.apply_expression(expr.EXCITED)
        self.ctx.transition(State.GREETING, self.face)
        # GREETING은 head_tracker가 이미 양보하지만, 명시적으로 lock도 잡아 안전.
        async with motion_busy_scope(self.ctx):
            try:
                while True:
                    if self.servos is not None:
                        await poses.dance(
                            self.servos, self.face,
                            bpm=int(bpm), beats=8,
                        )
                    else:
                        await asyncio.sleep(60.0 * 8 / bpm)
            except asyncio.CancelledError:
                log.info("🎵 음악 dance 종료 — 정리 중")
                raise
            finally:
                self.ctx.transition(
                    State.WATCHING if self.ctx.user_present else State.IDLE,
                    self.face,
                )

    async def run(self) -> None:
        monitor = AudioMonitor(
            self.mic,
            on_clap=self._on_clap,
            on_music_start=self._on_music_start,
            on_music_stop=self._on_music_stop,
        )
        await monitor.run()


async def run_audio_reactive(
    mic: Microphone,
    face: FaceState,
    ctx: StateContext,
    perception: PerceptionState | None = None,
    servos: ServoController | None = None,
) -> None:
    """task 진입점."""
    ar = AudioReactive(mic, face, ctx, perception=perception, servos=servos)
    await ar.run()
