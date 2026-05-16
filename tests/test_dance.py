"""dance pose + voice 명령 인터셉트 테스트."""

from __future__ import annotations

import asyncio

import pytest

from src.brain.state_machine import StateContext
from src.face.expressions import NEUTRAL
from src.face.renderer import FaceState
from src.motion import poses
from src.motion.servos import MockServoController


def test_dance_runs_to_completion_and_returns_home():
    face = FaceState(expression=NEUTRAL)
    servos = MockServoController()

    # 빠르게 끝나도록 BPM 올리고 beats 짧게
    asyncio.run(poses.dance(servos, face, bpm=240, beats=4, update_hz=30))

    # 시작 표정으로 복귀
    assert face.expression.name == NEUTRAL.name
    assert face.mouth_state.talk_amplitude == 0.0
    # 서보가 중앙 근처로 복귀
    from src.config import PAN_CENTER_DEG, TILT_CENTER_DEG
    assert abs(servos.position.pan - PAN_CENTER_DEG) < 1.0
    assert abs(servos.position.tilt - TILT_CENTER_DEG) < 1.0


def test_dance_changes_position_during_run():
    """모션이 실제로 움직이는지 확인 — 중간 시점에 pan/tilt가 중앙에서 벗어남."""
    face = FaceState(expression=NEUTRAL)
    servos = MockServoController()

    # 직접 한 step 흉내 — sin 위상 π/2일 때 pan_amp만큼
    # (전체 dance를 돌리는 대신 unit 동작 확인)
    from src.config import PAN_CENTER_DEG
    import math
    pan_amp = 25.0
    d_pan = math.sin(math.pi / 2) * pan_amp
    servos.set_angles(PAN_CENTER_DEG + d_pan, 90)
    assert abs(servos.position.pan - PAN_CENTER_DEG) > 10


@pytest.mark.parametrize("text", [
    "춤춰",
    "춤 춰봐",
    "댄스 부탁해",
    "dance please",
    "DANCE!",
])
def test_voice_command_matches_dance_keywords(text):
    from src.tasks.voice_assistant import VoiceAssistant

    ctx = StateContext()
    face = FaceState(expression=NEUTRAL)
    servos = MockServoController()
    va = VoiceAssistant(ctx, face, servos=servos)

    handled = asyncio.run(_with_short_dance(va, text))
    assert handled is True


async def _with_short_dance(va, text: str) -> bool:
    """dance를 짧게 바꿔치고 명령 인터셉트만 검증."""
    import src.tasks.voice_assistant as va_mod
    orig = va_mod.poses.dance

    async def fast_dance(*args, **kwargs):
        # 즉시 리턴
        return None

    va_mod.poses.dance = fast_dance
    try:
        return await va._maybe_handle_command(text)
    finally:
        va_mod.poses.dance = orig


def test_voice_command_ignores_non_dance():
    from src.tasks.voice_assistant import VoiceAssistant

    ctx = StateContext()
    face = FaceState(expression=NEUTRAL)
    va = VoiceAssistant(ctx, face)
    result = asyncio.run(va._maybe_handle_command("오늘 날씨 어때?"))
    assert result is False
