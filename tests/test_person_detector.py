"""Person detector 디바운스/히스테리시스 테스트."""

import time

import pytest

from src.sensors.base import SensorEventType
from src.vision.camera import Detection
from src.vision.person_detector import PersonDetector, PresenceState


def _person_det(conf: float = 0.9, bbox=(0.3, 0.2, 0.7, 0.9)) -> Detection:
    return Detection(class_id=1, class_name="person", confidence=conf, bbox=bbox)


def _other_det(name: str = "chair", conf: float = 0.7) -> Detection:
    return Detection(class_id=57, class_name=name, confidence=conf, bbox=(0, 0, 1, 1))


def test_initial_state_is_away():
    d = PersonDetector()
    assert d.state == PresenceState.AWAY


def test_single_detection_does_not_trigger():
    d = PersonDetector(confirm_frames=3)
    events = d.process([_person_det()])
    assert events == []
    assert d.state == PresenceState.AWAY


def test_confirmed_detection_emits_presence_new():
    d = PersonDetector(confirm_frames=3)
    for _ in range(3):
        events = d.process([_person_det()])
    assert d.state == PresenceState.PRESENT
    assert any(e.type == SensorEventType.PRESENCE_NEW for e in events)


def test_non_person_detection_does_not_count():
    d = PersonDetector(confirm_frames=3)
    for _ in range(5):
        d.process([_other_det()])
    assert d.state == PresenceState.AWAY


def test_low_confidence_does_not_count():
    d = PersonDetector(confirm_frames=2)
    d.process([_person_det(conf=0.3)])
    d.process([_person_det(conf=0.4)])
    assert d.state == PresenceState.AWAY


def test_away_emitted_after_timeout(monkeypatch):
    d = PersonDetector(confirm_frames=2, away_timeout_sec=1.0)
    # 등장
    d.process([_person_det()])
    d.process([_person_det()])
    assert d.state == PresenceState.PRESENT

    # 시간 점프
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 2.0)
    events = d.process([])
    assert d.state == PresenceState.AWAY
    assert any(e.type == SensorEventType.PRESENCE_LEFT for e in events)


def test_distance_reported_from_bbox_size():
    d = PersonDetector(confirm_frames=1)
    # 큰 bbox (가까운 사람)
    events_close = d.process([_person_det(bbox=(0.1, 0.1, 0.9, 0.9))])
    new_ev = next(e for e in events_close if e.type == SensorEventType.PRESENCE_NEW)
    distance_close = new_ev.data["distance_cm"]

    # reset
    d2 = PersonDetector(confirm_frames=1)
    events_far = d2.process([_person_det(bbox=(0.45, 0.45, 0.55, 0.55))])
    new_ev2 = next(e for e in events_far if e.type == SensorEventType.PRESENCE_NEW)
    distance_far = new_ev2.data["distance_cm"]

    # 작은 bbox일수록 멀어야 함
    assert distance_close < distance_far
