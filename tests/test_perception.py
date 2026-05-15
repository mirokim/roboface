"""PerceptionState 단위 테스트."""

from src.brain.perception import PerceptionState


def test_initial_state():
    p = PerceptionState()
    assert p.person_present is False
    assert p.person_bbox is None
    assert p.person_distance_cm == -1.0


def test_update_person_sets_state():
    p = PerceptionState()
    p.update_person(bbox=(0.3, 0.2, 0.7, 0.9), distance_cm=120)
    assert p.person_present is True
    assert p.person_bbox == (0.3, 0.2, 0.7, 0.9)
    assert p.person_distance_cm == 120
    assert p.last_person_seen_at > 0


def test_clear_person_resets():
    p = PerceptionState()
    p.update_person(bbox=(0.3, 0.2, 0.7, 0.9))
    p.clear_person()
    assert p.person_present is False
    assert p.person_bbox is None


def test_bbox_center():
    p = PerceptionState()
    assert p.person_bbox_center == (0.5, 0.5)  # default
    p.update_person(bbox=(0.2, 0.4, 0.8, 0.6))
    cx, cy = p.person_bbox_center
    assert abs(cx - 0.5) < 1e-6
    assert abs(cy - 0.5) < 1e-6


def test_bbox_center_offset():
    p = PerceptionState()
    p.update_person(bbox=(0.0, 0.0, 0.4, 0.4))  # 좌상단
    cx, cy = p.person_bbox_center
    assert abs(cx - 0.2) < 1e-6
    assert abs(cy - 0.2) < 1e-6
