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
        # run()에서 AudioMonitor 인스턴스 보관 — _music_dance_loop이 pause 호출용
        self._monitor: AudioMonitor | None = None

    # ─── 이벤트 핸들러 (audio_monitor 콜백) ───
    # 콜백은 마이크 백그라운드 스레드의 asyncio loop에서 호출됨 (audio_monitor가
    # asyncio.run loop 안에 있으므로 동일 loop). 그래서 직접 task 만들어도 OK.

    def _on_clap(self) -> None:
        # 콜백 진입 자체 로깅 — "왜 반응 없지" 진단 시 audio_monitor는 박수 잡았는데
        # 표정 변화가 없는 케이스(state 차단/덮어쓰기 등) 명확히 구분.
        log.info(f"👏 박수 콜백 발동 (state={self.ctx.state.value})")
        # 대화 중엔 양보
        if self.ctx.state in (State.TALKING, State.LISTENING):
            log.info(f"  → state={self.ctx.state.value}, 박수 무시")
            return
        # 표정만 — 끄덕 모션 제거 (시끄러우면 산만).
        # 0.5초는 LCD 안 보고 있으면 못 알아챔 → 1.5초로 늘림. 박수 N번 치면
        # 같은 표정으로 그 사이 계속 유지(flash_expression이 새 발동마다 교체).
        flash_expression(self.face, expr.SURPRISED, 1.5)
        log.info(f"  → SURPRISED flash 1.5s (이전 표정={self.face.expression.name})")

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
        """음악 끝날 때까지 dance 반복. cancel되면 정리.

        중요: dance 중 audio_monitor pause — 서보 소리가 onset으로 잡혀
        music_playing 유지 → 무한 dance 루프 차단. dance 끝나면 resume +
        baseline 회복 시간(1초) 부여 후 _onsets buffer clear됨(set_paused).
        """
        # BPM 범위 클램프
        bpm = max(70.0, min(160.0, bpm))
        self.face.apply_expression(expr.EXCITED)
        self.ctx.transition(State.GREETING, self.face)
        if self._monitor is not None:
            self._monitor.set_paused(True)
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
                # 서보 노이즈 잔향 가시면 baseline 안정 후 resume — 짧게 1초.
                if self._monitor is not None:
                    await asyncio.sleep(1.0)
                    self._monitor.set_paused(False)

    async def run(self) -> None:
        self._monitor = AudioMonitor(
            self.mic,
            on_clap=self._on_clap,
            on_music_start=self._on_music_start,
            on_music_stop=self._on_music_stop,
        )
        await self._monitor.run()


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
