"""robot_stats — 이벤트/decay/회복 + 표정 추천 테스트."""

from __future__ import annotations

import time

import pytest

from src.brain import memory, stats as robot_stats


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """각 테스트마다 빈 DB로 격리."""
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("src.brain.memory.DB_PATH", db_path)
    monkeypatch.setattr("src.brain.memory._DB_INITIALIZED", False)
    # stats 모듈의 캐시 reset
    monkeypatch.setattr(robot_stats, "_STATS", None)
    memory.init_db(db_path)
    yield


def test_initial_defaults():
    s = robot_stats.get()
    assert 0 <= s.energy <= 100
    assert 0 <= s.mood <= 100
    assert s.energy > 50  # 시작 시 충분


def test_event_increases_mood():
    s = robot_stats.get()
    mood_before = s.mood
    robot_stats.on_event("wave")
    s = robot_stats.get()
    assert s.mood > mood_before


def test_negative_event():
    robot_stats.get()  # init
    robot_stats.on_event("thumb_down")
    s = robot_stats.get()
    # mood 줄어들었는지 (시작값 65 - 3 = 62 근처)
    assert s.mood < 65


def test_clamp_max():
    s = robot_stats.get()
    s.mood = 200
    s.clamp()
    assert s.mood == 100


def test_clamp_min():
    s = robot_stats.get()
    s.mood = -10
    s.clamp()
    assert s.mood == 0


def test_low_energy_suggests_sleepy(monkeypatch):
    s = robot_stats.get()
    s.energy = 20.0
    monkeypatch.setattr(robot_stats, "_save", lambda s: None)
    assert robot_stats.suggested_expression() == "SLEEPY"


def test_high_mood_suggests_happy(monkeypatch):
    s = robot_stats.get()
    s.energy = 90.0
    s.mood = 90.0
    s.social = 80.0
    s.curiosity = 80.0
    monkeypatch.setattr(robot_stats, "_save", lambda s: None)
    assert robot_stats.suggested_expression() == "HAPPY"


def test_unknown_event_noop():
    before = robot_stats.get().mood
    robot_stats.on_event("nonexistent_event")
    assert robot_stats.get().mood == pytest.approx(before, abs=0.5)


def test_presence_tick_recovers_social():
    s = robot_stats.get()
    s.social = 30.0
    robot_stats._save(s)
    robot_stats.on_presence_tick(3600.0)  # 1 시간
    s = robot_stats.get()
    assert s.social > 30   # 회복됨


def test_mood_label_categories():
    s = robot_stats.get()
    s.energy = 10.0
    robot_stats._save(s)
    assert robot_stats.mood_label() == "졸림"
