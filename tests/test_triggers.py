"""트리거 평가기 단위 테스트."""

import time
from unittest.mock import patch

import pytest

from src.brain import triggers
from src.brain.state_machine import State, StateContext
from src.config import BEHAVIOR


def test_quiet_hours_logic():
    from datetime import datetime
    night = datetime(2026, 5, 11, 23, 0)
    day = datetime(2026, 5, 11, 14, 0)
    assert triggers._is_quiet_hours(night) is True
    assert triggers._is_quiet_hours(day) is False


def test_greeting_only_when_user_just_appeared():
    ctx = StateContext(state=State.WATCHING, user_present=True,
                       last_user_seen_at=time.time())
    g = triggers.check_greeting(ctx)
    assert g is not None
    assert g.kind == "greeting"


def test_greeting_skipped_when_already_greeting():
    ctx = StateContext(state=State.GREETING, user_present=True,
                       last_user_seen_at=time.time())
    assert triggers.check_greeting(ctx) is None


def test_proactive_blocked_when_user_absent():
    ctx = StateContext(state=State.IDLE, user_present=False)
    assert triggers._proactive_allowed(ctx) is False


def test_evaluate_all_returns_sorted_by_priority():
    ctx = StateContext(state=State.WATCHING, user_present=True,
                       last_user_seen_at=time.time())
    triggers_list = triggers.evaluate_all(ctx)
    if len(triggers_list) > 1:
        assert all(triggers_list[i].priority >= triggers_list[i + 1].priority
                   for i in range(len(triggers_list) - 1))
