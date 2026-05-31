"""head_tracker 동작 — 특히 motion 양보 복귀 시 servo 위치 resync.

회귀 가드: poses.sway/dance 후 head_tracker가 자신의 stale 내부 상태
(pan_current)로 명령하면 실제 서보 위치(다른 모션이 옮긴 위치)에서
큰 점프 → "휙" 발생. resync 로직이 이 점프를 막아야 함.
"""

from __future__ import annotations

import asyncio

import pytest

from src.brain.perception import PerceptionState
from src.brain.state_machine import State, StateContext
from src.config import PAN_CENTER_DEG, TILT_CENTER_DEG
from src.motion.servos import MockServoController
from src.tasks import head_tracker


@pytest.mark.asyncio
async def test_head_tracker_resyncs_after_motion_busy():
    """ctx.motion_busy=True 동안 외부 모션이 서보 위치를 바꿔도, 풀린 뒤
    head_tracker가 자신의 pan_current를 실제 위치로 resync해서 갑자기 큰
    점프 명령을 보내지 않아야 한다.
    """
    servos = MockServoController()
    perception = PerceptionState()
    ctx = StateContext()
    ctx.state = State.WATCHING
    ctx.user_present = True
    # 사용자가 정면 가까이 있는 것으로 가정 (bbox center ≈ 0.5, 0.5)
    perception.person_present = True
    perception.person_bbox = (0.4, 0.4, 0.6, 0.6)

    task = asyncio.create_task(head_tracker.run_head_tracker(servos, perception, ctx))
    try:
        # 1) 처음 잠깐 돌려서 정상 흐름 진입
        await asyncio.sleep(0.3)

        # 2) 양보 — 다른 모션이 서보를 옮겼다고 가정 (예: 인사 끄덕)
        ctx.motion_busy = True
        await asyncio.sleep(0.2)
        # 외부 모션이 옆으로 옮김
        off_pan = PAN_CENTER_DEG - 20.0
        off_tilt = TILT_CENTER_DEG + 8.0
        servos.set_angles(off_pan, off_tilt)
        assert servos.position.pan == pytest.approx(off_pan)
        await asyncio.sleep(0.2)

        # 3) 양보 해제 — head_tracker가 resync 후 다음 명령은 작은 delta여야
        ctx.motion_busy = False
        # resync 발동 cycle (period = 1/15s) + 한두 cycle 추가
        await asyncio.sleep(0.4)

        # 사용자 위치 변화 없음 → head_tracker가 큰 명령 보낼 이유 없음.
        # 실제 서보 위치가 off에서 크게 벗어나지 않아야 함 (resync 안 했으면
        # CENTER 쪽으로 휙 갔을 것).
        pan_drift = abs(servos.position.pan - off_pan)
        tilt_drift = abs(servos.position.tilt - off_tilt)
        assert pan_drift < 15.0, (
            f"resync 안 됨 — pan이 {off_pan}→{servos.position.pan} ({pan_drift:.1f}° 점프)"
        )
        assert tilt_drift < 10.0, (
            f"resync 안 됨 — tilt가 {off_tilt}→{servos.position.tilt} ({tilt_drift:.1f}° 점프)"
        )
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
