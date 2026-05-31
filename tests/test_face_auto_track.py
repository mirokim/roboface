"""FaceMemory 자동 학습 (auto_track + get_owner)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.vision.face_memory import (
    AUTO_NAME_PREFIX,
    DEFAULT_OWNER_MIN_SEEN,
    FaceMemory,
)


def _img(seed: int, size: int = 80):
    """동일 seed면 동일 이미지 — 같은 사람 시뮬레이션."""
    return np.random.RandomState(seed).randint(0, 255, (size, size), dtype=np.uint8)


def test_auto_track_new_face_creates_auto_cluster(tmp_path):
    fm = FaceMemory(tmp_path / "f.db")
    img = _img(1)
    result = fm.auto_track(img)
    assert result is not None
    name, count = result
    assert name.startswith(AUTO_NAME_PREFIX)
    assert count == 1
    assert name in fm.list_names()


def test_auto_track_matching_face_increments_count(tmp_path, monkeypatch):
    """같은 사람 반복 시 seen_count 누적 (throttle 통과해야)."""
    # throttle 0초로 — 즉시 카운트
    import src.vision.face_memory as fm_mod
    monkeypatch.setattr(fm_mod, "_AUTO_COUNT_THROTTLE_SEC", 0.0)

    fm = FaceMemory(tmp_path / "f.db")
    img = _img(1)
    for _ in range(5):
        result = fm.auto_track(img)
        assert result is not None

    # 5회 후 같은 cluster + seen_count >= 5
    name, count = fm.auto_track(img)
    assert count >= 5


def test_auto_track_throttle_prevents_rapid_count(tmp_path):
    """기본 throttle(10s) 안엔 카운트 한 번만."""
    fm = FaceMemory(tmp_path / "f.db")
    img = _img(1)
    for _ in range(5):
        fm.auto_track(img)
    # 5번 호출했지만 throttle 안이라 count는 1만
    _, count = fm.auto_track(img)
    assert count == 1


def test_auto_track_different_faces_create_separate_clusters(tmp_path):
    fm = FaceMemory(tmp_path / "f.db")
    n1, _ = fm.auto_track(_img(1))
    n2, _ = fm.auto_track(_img(42))
    assert n1 != n2
    assert len(fm.list_names()) == 2


def test_get_owner_returns_most_seen(tmp_path, monkeypatch):
    """seen_count 가장 많은 cluster를 owner로."""
    import src.vision.face_memory as fm_mod
    monkeypatch.setattr(fm_mod, "_AUTO_COUNT_THROTTLE_SEC", 0.0)

    fm = FaceMemory(tmp_path / "f.db")
    # 사람 A: 25번 (owner 후보)
    img_a = _img(1)
    for _ in range(25):
        fm.auto_track(img_a)
    # 사람 B: 5번 (guest)
    img_b = _img(42)
    for _ in range(5):
        fm.auto_track(img_b)

    owner = fm.get_owner(min_seen=20)
    assert owner is not None
    name, count = owner
    assert count >= 25
    # owner는 A의 cluster이어야 — img_a embedding과 매칭
    a_track = fm.auto_track(img_a)
    assert name == a_track[0]


def test_get_owner_returns_none_below_threshold(tmp_path):
    """min_seen 미만이면 owner 없음."""
    fm = FaceMemory(tmp_path / "f.db")
    fm.auto_track(_img(1))   # seen=1
    owner = fm.get_owner(min_seen=DEFAULT_OWNER_MIN_SEEN)
    assert owner is None


def test_get_owner_empty_db(tmp_path):
    fm = FaceMemory(tmp_path / "f.db")
    assert fm.get_owner() is None


def test_explicit_register_compatible_with_auto_track(tmp_path):
    """explicit register name(미로)도 auto_track으로 매칭되면 count++."""
    import src.vision.face_memory as fm_mod
    import unittest.mock as mock
    with mock.patch.object(fm_mod, "_AUTO_COUNT_THROTTLE_SEC", 0.0):
        fm = FaceMemory(tmp_path / "f.db")
        img = _img(1)
        fm.register("미로", img)
        # 같은 얼굴 auto_track — auto cluster 생성 X, 기존 "미로" cluster count++
        result = fm.auto_track(img)
        assert result is not None
        name, count = result
        assert name == "미로"
        assert not name.startswith(AUTO_NAME_PREFIX)


def test_migration_adds_seen_count_column(tmp_path):
    """이전 schema(seen_count 없음) DB 열면 ALTER로 마이그레이션."""
    import sqlite3
    db = tmp_path / "old.db"
    # 옛 schema 수동 생성
    conn = sqlite3.connect(str(db))
    conn.execute("""
        CREATE TABLE faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            embedding BLOB NOT NULL,
            created_at REAL NOT NULL,
            last_seen_at REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()

    # FaceMemory 열면 ALTER 실행
    fm = FaceMemory(db)
    # seen_count 컬럼이 추가됐어야
    cur = fm.conn.execute("PRAGMA table_info(faces)")
    cols = {row[1] for row in cur.fetchall()}
    assert "seen_count" in cols
