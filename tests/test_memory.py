"""SQLite 메모리 모듈 단위 테스트."""

import time

import pytest

from src.brain import memory


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """각 테스트마다 임시 DB."""
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(memory, "DB_PATH", db_path)
    monkeypatch.setattr(memory, "_DB_INITIALIZED", False)
    memory.init_db(db_path)
    yield


def test_work_session_lifecycle():
    sid = memory.start_work_session()
    assert sid > 0
    time.sleep(0.05)
    memory.end_work_session(sid)


def test_today_total_seconds_starts_zero():
    assert memory.today_total_seconds() == 0


def test_proactive_log_and_count():
    assert memory.proactive_count_last_hour() == 0
    memory.log_proactive("greeting", "안녕!")
    assert memory.proactive_count_last_hour() == 1


def test_schedule_add_and_query():
    sid = memory.add_schedule("meeting", "2026-05-15T14:00", "Test meeting", 0.9)
    assert sid > 0
    pending = memory.unsynced_schedules()
    assert any(s.id == sid for s in pending)
    memory.mark_schedule_synced(sid)
    pending2 = memory.unsynced_schedules()
    assert not any(s.id == sid for s in pending2)


def test_user_pattern_kv():
    memory.set_pattern("test_key", {"value": 42})
    assert memory.get_pattern("test_key") == {"value": 42}
    assert memory.get_pattern("missing", default="x") == "x"


# === learned_facts ===

def test_remember_fact_default_category_source():
    memory.remember_fact("좋아하는음료", "라떼")
    facts = memory.all_facts()
    f = next(f for f in facts if f["key"] == "좋아하는음료")
    assert f["value"] == "라떼"
    assert f["category"] == "user"
    assert f["source"] == "agent"
    assert f["last_used_at"] is None


def test_seed_robot_facts_category_self():
    added = memory.seed_robot_facts()
    assert added > 0
    facts = memory.all_facts(limit=50)
    self_facts = [f for f in facts if f["category"] == "self"]
    assert len(self_facts) >= 6
    assert all(f["source"] == "seed" for f in self_facts)


def test_mark_facts_used_updates_timestamp():
    memory.remember_fact("k1", "v1")
    memory.remember_fact("k2", "v2")
    before = memory.all_facts()
    assert all(f["last_used_at"] is None for f in before)
    memory.mark_facts_used(["k1", "k2"])
    after = memory.all_facts()
    assert all(f["last_used_at"] is not None for f in after)


def test_mark_facts_used_empty_noop():
    memory.mark_facts_used([])   # 빈 리스트 — 에러 X


@pytest.mark.parametrize("key,value,expected", [
    ("내일 9시 치과 예약", "오전 9시", True),
    ("프로젝트 마감", "다음달 15일", True),
    ("금요일 저녁 약속", "친구와", True),
    ("내일 오후 3시 회의", "김 부장님과의 회의", True),
    ("좋아하는음료", "라떼", False),
    ("관심분야", "독서", False),
    ("이름", "지호", False),
    # 시간 단어만 있고 schedule 키워드 없으면 fact로 — agent 자의적 분류 방지
    ("취미", "내일 등산", False),
])
def test_is_schedule_like(key, value, expected):
    assert memory.is_schedule_like(key, value) is expected
