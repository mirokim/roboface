"""agent _do_speak hard gate — 호명 없는 사용자 발화엔 응답 X."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest


def _make_agent(perception_speech_at=0.0, perception_called_at=0.0,
                last_speak_at=0.0):
    """RobotAgent stub — _do_speak에 필요한 attr만 set."""
    from src.brain.agent import RobotAgent
    from src.brain.state_machine import StateContext

    agent = RobotAgent.__new__(RobotAgent)   # __init__ 우회
    agent.face = MagicMock()
    agent.ctx = StateContext()
    agent.ctx.user_present = True
    agent.perception = MagicMock()
    agent.perception.last_user_speech_at = perception_speech_at
    agent.perception.last_user_called_at = perception_called_at
    agent._last_speak_at = last_speak_at
    agent._post_speak_baseline = None
    agent._last_speak_text = None
    return agent


def test_skip_when_user_spoke_without_calling():
    """사용자가 일반 발화 (호명 X) → _do_speak skip — 응답 차단."""
    now = time.time()
    agent = _make_agent(
        perception_speech_at=now - 2.0,   # 2초 전 발화 (recent)
        perception_called_at=0.0,          # 호명 한 적 없음
    )
    with patch("src.brain.agent.memory.log_robot"):
        asyncio.run(agent._do_speak({"text": "응 안녕"}))
    # 발화 안 함 — _last_speak_at 안 갱신
    assert agent._last_speak_at == 0.0


def test_responds_when_called_recently():
    """호명 후 → _do_speak 실행."""
    now = time.time()
    agent = _make_agent(
        perception_speech_at=now - 1.0,
        perception_called_at=now - 1.0,    # 방금 호명
    )
    # asyncio.create_task 안 호출되게 fake_speak도 모킹
    with patch("src.brain.agent.memory.log_robot"), \
         patch("src.audio.fake_tts.speak"):
        asyncio.run(agent._do_speak({"text": "응?"}))
    # _last_speak_at 갱신 = 발화 실행됨
    assert agent._last_speak_at > 0


def test_responds_within_60s_window_even_without_recall():
    """호명 후 60초 윈도우 안엔 추가 발화도 응답 OK (대화 흐름)."""
    now = time.time()
    agent = _make_agent(
        perception_speech_at=now - 1.0,    # 방금 일반 발화
        perception_called_at=now - 30.0,   # 30초 전 호명 (윈도우 안)
    )
    with patch("src.brain.agent.memory.log_robot"), \
         patch("src.audio.fake_tts.speak"):
        asyncio.run(agent._do_speak({"text": "그래"}))
    assert agent._last_speak_at > 0


def test_skip_after_60s_window_closed():
    """호명 후 60s 지나면 다시 침묵 모드 — 호명 없는 발화엔 응답 X."""
    now = time.time()
    agent = _make_agent(
        perception_speech_at=now - 1.0,
        perception_called_at=now - 120.0,  # 2분 전 호명 (윈도우 밖)
    )
    with patch("src.brain.agent.memory.log_robot"):
        asyncio.run(agent._do_speak({"text": "응 안녕"}))
    assert agent._last_speak_at == 0.0


def test_proactive_speech_no_user_input_still_cooldown_limited():
    """proactive(사용자 발화 없는) speak는 호명 가드 통과 — cooldown만 적용."""
    now = time.time()
    agent = _make_agent(
        perception_speech_at=0.0,         # 사용자 발화 X
        perception_called_at=0.0,
        last_speak_at=now - 5.0,           # 마지막 발화 5초 전 (cooldown 90s 안)
    )
    with patch("src.brain.agent.memory.log_robot"):
        asyncio.run(agent._do_speak({"text": "음, 조용하네"}))
    # cooldown으로 skip — _last_speak_at 안 갱신
    assert agent._last_speak_at == now - 5.0


def test_proactive_speech_after_cooldown_executes():
    """proactive — cooldown 지나면 실행."""
    now = time.time()
    agent = _make_agent(
        perception_speech_at=0.0,
        perception_called_at=0.0,
        last_speak_at=now - 200.0,         # 마지막 발화 200초 전 (cooldown 통과)
    )
    with patch("src.brain.agent.memory.log_robot"), \
         patch("src.audio.fake_tts.speak"):
        asyncio.run(agent._do_speak({"text": "어, 안녕"}))
    assert agent._last_speak_at > now - 200.0
