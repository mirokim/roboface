"""Phase A-D 추가 기능 단위 테스트 — agent 컨텍스트 보강, recall, remember_fact."""

from __future__ import annotations

import time
from datetime import datetime

import pytest

from src.brain import agent, memory
from src.brain.perception import PerceptionState
from src.brain.state_machine import StateContext


# === memory.search_conversation ===

def test_search_conversation_finds_keyword(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(memory, "DB_PATH", db_path)
    memory._DB_INITIALIZED = False
    memory.init_db(db_path)

    memory.log_user("오늘 라떼 마셨어")
    memory.log_user("점심 메뉴 뭐 먹지")
    memory.log_robot("좋은 아침")

    rows = memory.search_conversation("라떼", days=1)
    assert len(rows) == 1
    assert "라떼" in rows[0]["text"]

    rows2 = memory.search_conversation("없는키워드", days=1)
    assert rows2 == []


def test_search_conversation_empty_keyword_returns_empty(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(memory, "DB_PATH", db_path)
    memory._DB_INITIALIZED = False
    memory.init_db(db_path)

    assert memory.search_conversation("") == []
    assert memory.search_conversation("   ") == []


# === memory.remember_fact / all_facts ===

def test_remember_fact_roundtrip(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(memory, "DB_PATH", db_path)
    memory._DB_INITIALIZED = False
    memory.init_db(db_path)

    memory.remember_fact("좋아하는음료", "라떼")
    memory.remember_fact("취미", "독서")

    facts = memory.all_facts()
    assert len(facts) == 2
    keys = {f["key"] for f in facts}
    assert keys == {"좋아하는음료", "취미"}


def test_remember_fact_updates_existing(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(memory, "DB_PATH", db_path)
    memory._DB_INITIALIZED = False
    memory.init_db(db_path)

    memory.remember_fact("좋아하는음료", "라떼")
    memory.remember_fact("좋아하는음료", "아메리카노")
    facts = memory.all_facts()
    assert len(facts) == 1
    assert facts[0]["value"] == "아메리카노"


def test_remember_fact_ignores_empty(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(memory, "DB_PATH", db_path)
    memory._DB_INITIALIZED = False
    memory.init_db(db_path)

    memory.remember_fact("", "값")
    memory.remember_fact("키", "")
    assert memory.all_facts() == []


# === memory.last_proactive_log ===

def test_last_proactive_log(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(memory, "DB_PATH", db_path)
    memory._DB_INITIALIZED = False
    memory.init_db(db_path)

    assert memory.last_proactive_log() is None
    memory.log_proactive("greeting", "안녕")
    last = memory.last_proactive_log()
    assert last is not None
    assert last["trigger"] == "greeting"
    assert last["message"] == "안녕"


# === agent._time_hint ===

def test_time_hint_lunch():
    h = agent._time_hint(datetime(2026, 5, 23, 12, 15))
    assert h is not None and "점심" in h


def test_time_hint_afternoon_slump():
    h = agent._time_hint(datetime(2026, 5, 23, 15, 0))
    assert h is not None and "오후" in h


def test_time_hint_none_for_normal_hour():
    # 평범한 오전 시간대 (9~11시) + 분 3분 이후 — 특별한 힌트 없음
    assert agent._time_hint(datetime(2026, 5, 23, 10, 30)) is None


def test_time_hint_hour_top():
    # 정시 0~2분 — "방금 N시 정각" 표시
    h = agent._time_hint(datetime(2026, 5, 23, 10, 1))
    assert h is not None and "10시" in h


# === agent._build_situation — crash 안전성 ===

def test_build_situation_minimal_no_crash(tmp_path, monkeypatch):
    """모든 필드 비어 있어도 _build_situation이 크래시 X."""
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(memory, "DB_PATH", db_path)
    memory._DB_INITIALIZED = False
    memory.init_db(db_path)

    ctx = StateContext()
    perception = PerceptionState()
    text = agent._build_situation(ctx, perception, work_minutes=None)
    assert isinstance(text, str)
    assert "현재 시각" in text
    assert "사용자 존재" in text


def test_build_situation_includes_current_emotion(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(memory, "DB_PATH", db_path)
    memory._DB_INITIALIZED = False
    memory.init_db(db_path)

    ctx = StateContext(user_present=True)
    perception = PerceptionState(
        current_emotion="smile", current_emotion_at=time.time(),
    )
    text = agent._build_situation(ctx, perception, work_minutes=10.0)
    assert "smile" in text


def test_build_situation_skips_neutral_emotion(tmp_path, monkeypatch):
    """neutral은 정보 없음 — 프롬프트에 포함 X."""
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(memory, "DB_PATH", db_path)
    memory._DB_INITIALIZED = False
    memory.init_db(db_path)

    ctx = StateContext(user_present=True)
    perception = PerceptionState(
        current_emotion="neutral", current_emotion_at=time.time(),
    )
    text = agent._build_situation(ctx, perception, None)
    assert "사용자 표정" not in text


def test_build_situation_skips_stale_emotion(tmp_path, monkeypatch):
    """90초 넘은 표정은 무시."""
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(memory, "DB_PATH", db_path)
    memory._DB_INITIALIZED = False
    memory.init_db(db_path)

    ctx = StateContext(user_present=True)
    perception = PerceptionState(
        current_emotion="smile", current_emotion_at=time.time() - 200,
    )
    text = agent._build_situation(ctx, perception, None)
    assert "사용자 표정" not in text


def test_build_situation_includes_facts(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(memory, "DB_PATH", db_path)
    memory._DB_INITIALIZED = False
    memory.init_db(db_path)

    memory.remember_fact("좋아하는음료", "라떼")
    ctx = StateContext(user_present=True)
    text = agent._build_situation(ctx, PerceptionState(), None)
    assert "학습한 사실" in text
    assert "라떼" in text
