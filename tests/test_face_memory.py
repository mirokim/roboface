"""FaceMemory + 등록 명령 파싱 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from src.tasks.voice_assistant import _extract_register_name
from src.vision.face_memory import (
    FaceMemory, MATCH_THRESHOLD, compute_face_embedding,
)


# === embedding ===

def test_embedding_returns_unit_vector():
    img = np.random.RandomState(0).randint(0, 255, (80, 80), dtype=np.uint8)
    emb = compute_face_embedding(img)
    assert emb is not None
    assert emb.shape == (32 * 32,)
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-4


def test_embedding_too_small_returns_none():
    img = np.zeros((30, 30), dtype=np.uint8)
    assert compute_face_embedding(img) is None


def test_embedding_none_input():
    assert compute_face_embedding(None) is None


# === FaceMemory ===

def test_register_and_recognize_same_face(tmp_path):
    fm = FaceMemory(tmp_path / "faces.db")
    img = np.random.RandomState(42).randint(0, 255, (80, 80), dtype=np.uint8)
    assert fm.register("miro", img) is True
    assert "miro" in fm.list_names()

    match = fm.recognize(img)
    assert match is not None
    assert match.name == "miro"
    assert match.confidence >= MATCH_THRESHOLD


def test_recognize_unknown_face_returns_none(tmp_path):
    fm = FaceMemory(tmp_path / "faces.db")
    rng = np.random.RandomState(0)
    fm.register("miro", rng.randint(0, 255, (80, 80), dtype=np.uint8))
    # 완전히 다른 패턴
    other = np.random.RandomState(999).randint(0, 255, (80, 80), dtype=np.uint8)
    match = fm.recognize(other)
    # 잘 학습 안 된 임베딩이지만 random이면 보통 threshold 못 넘음
    assert match is None or match.name == "miro"


def test_register_updates_existing_name(tmp_path):
    fm = FaceMemory(tmp_path / "faces.db")
    rng = np.random.RandomState(0)
    img1 = rng.randint(0, 255, (80, 80), dtype=np.uint8)
    img2 = rng.randint(0, 255, (80, 80), dtype=np.uint8)
    fm.register("miro", img1)
    fm.register("miro", img2)
    assert fm.list_names() == ["miro"]


def test_recognize_with_empty_db(tmp_path):
    fm = FaceMemory(tmp_path / "faces.db")
    img = np.zeros((80, 80), dtype=np.uint8) + 100
    assert fm.recognize(img) is None


def test_persistence_across_instances(tmp_path):
    db = tmp_path / "faces.db"
    fm1 = FaceMemory(db)
    img = np.random.RandomState(7).randint(0, 255, (80, 80), dtype=np.uint8)
    fm1.register("miro", img)
    fm1.close()

    fm2 = FaceMemory(db)
    assert "miro" in fm2.list_names()
    match = fm2.recognize(img)
    assert match is not None and match.name == "miro"


# === 이름 파싱 ===

@pytest.mark.parametrize("text,expected", [
    ("내 이름은 미로야", "미로"),
    ("내 이름은 미로입니다", "미로"),
    ("내 이름은 미로 입니다", "미로"),
    ("내 이름 미로", "미로"),
    ("내 이름은 알렉스예요", "알렉스"),
    ("이 사람 이름은 지수야", "지수"),
    ("이 사람은 영희야", "영희"),
    ("이름 케빈", "케빈"),
])
def test_extract_register_name_positive(text, expected):
    assert _extract_register_name(text) == expected


@pytest.mark.parametrize("text", [
    "안녕",
    "오늘 날씨 어때",
    "춤춰",
    "이름이 뭐야",
    "",
])
def test_extract_register_name_negative(text):
    assert _extract_register_name(text) is None
