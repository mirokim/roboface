"""HandsUpDetector / HeadNodDetector / HeadShakeDetector 테스트."""

from __future__ import annotations

import math

import numpy as np

from src.vision.camera import (
    KP_L_EAR, KP_L_EYE, KP_L_SHOULDER, KP_L_WRIST, KP_NOSE,
    KP_R_EAR, KP_R_EYE, KP_R_SHOULDER, KP_R_WRIST,
)
from src.vision.pose_gestures import (
    GazeAtMeDetector, HandsUpDetector, HeadNodDetector, HeadShakeDetector,
    face_orientation,
)


def _kps(
    *,
    nose_x: float = 0.5, nose_y: float = 0.3,
    l_wrist_x: float = 0.4, l_wrist_y: float = 0.5,
    r_wrist_x: float = 0.6, r_wrist_y: float = 0.5,
    shoulder_l_x: float = 0.4, shoulder_r_x: float = 0.6,
    shoulder_y: float = 0.4,
    l_eye_x: float = 0.46, r_eye_x: float = 0.54,
    eye_y: float = 0.26,
    conf: float = 0.9,
) -> np.ndarray:
    arr = np.zeros((17, 3), dtype=np.float32)
    arr[KP_NOSE] = [nose_x, nose_y, conf]
    arr[KP_L_EYE] = [l_eye_x, eye_y, conf]
    arr[KP_R_EYE] = [r_eye_x, eye_y, conf]
    arr[KP_L_SHOULDER] = [shoulder_l_x, shoulder_y, conf]
    arr[KP_R_SHOULDER] = [shoulder_r_x, shoulder_y, conf]
    arr[KP_L_WRIST] = [l_wrist_x, l_wrist_y, conf]
    arr[KP_R_WRIST] = [r_wrist_x, r_wrist_y, conf]
    return arr


# ─── HandsUpDetector ───
# 어깨 기준 좌표를 쓰는 새 로직 (카메라가 낮아서 코가 화면 위쪽에 붙는 경우
# 대비). threshold = shoulder_y - shoulder_width * 0.3.
# 기본 _kps: shoulder_y=0.4, shoulder_width=0.2 → threshold=0.34.

def test_hands_up_detected_after_hold():
    """양 손목이 어깨보다 위로 (어깨너비×0.3 만큼) 충분히 올라가면 감지."""
    det = HandsUpDetector(fps=5.0, hold_sec=0.6)
    # 양손이 어깨 위로 충분히 (0.2 < 0.34)
    kps = _kps(shoulder_y=0.4, l_wrist_y=0.2, r_wrist_y=0.2)
    hits = 0
    for _ in range(5):
        if det.process(kps):
            hits += 1
    assert hits == 1   # 한 번 잡히면 cooldown


def test_hands_up_not_detected_when_only_one_hand_up():
    det = HandsUpDetector(fps=5.0, hold_sec=0.6)
    # 왼손만 위, 오른손은 어깨 아래
    kps = _kps(shoulder_y=0.4, l_wrist_y=0.2, r_wrist_y=0.7)
    for _ in range(10):
        assert not det.process(kps)


def test_hands_up_not_detected_when_both_just_below_threshold():
    """어깨보다 살짝 위지만 어깨너비×0.3 만큼은 안 올라간 경우 — 감지 X."""
    det = HandsUpDetector(fps=5.0, hold_sec=0.6)
    # threshold=0.34인데 wrist=0.36 → 아직 안 들어옴
    kps = _kps(shoulder_y=0.4, l_wrist_y=0.36, r_wrist_y=0.36)
    for _ in range(10):
        assert not det.process(kps)


def test_hands_up_not_detected_when_both_below_shoulder():
    """양손이 어깨 아래 — 감지 X. 카메라 시점 무관."""
    det = HandsUpDetector(fps=5.0, hold_sec=0.6)
    kps = _kps(shoulder_y=0.4, l_wrist_y=0.5, r_wrist_y=0.5)
    for _ in range(10):
        assert not det.process(kps)


def test_hands_up_detected_with_low_camera_angle():
    """낮은 카메라 시점 — 코가 화면 위쪽(0.15), 어깨가 중간(0.5)에 있어도
    손이 어깨 위로 잘 올라가면 감지. 기존 nose-기준 로직은 실패했을 케이스."""
    det = HandsUpDetector(fps=5.0, hold_sec=0.6)
    # 어깨=0.5, threshold=0.5-0.06=0.44. 손목=0.3은 코(0.15)보다 아래지만
    # 어깨보다는 확실히 위 → 감지 OK.
    kps = _kps(
        nose_y=0.15, eye_y=0.12,
        shoulder_y=0.5, l_wrist_y=0.3, r_wrist_y=0.3,
    )
    hits = sum(1 for _ in range(5) if det.process(kps))
    assert hits == 1


def test_hands_up_cooldown():
    det = HandsUpDetector(fps=5.0, hold_sec=0.6, cooldown_sec=10.0)
    kps = _kps(shoulder_y=0.4, l_wrist_y=0.2, r_wrist_y=0.2)
    hits = sum(1 for _ in range(20) if det.process(kps))
    assert hits == 1


def test_hands_up_resets_on_none():
    det = HandsUpDetector(fps=5.0, hold_sec=0.6)
    kps_up = _kps(shoulder_y=0.4, l_wrist_y=0.2, r_wrist_y=0.2)
    det.process(kps_up)
    det.process(kps_up)
    # None 들어오면 consecutive 리셋
    det.process(None)
    # 다시 시작 — 1프레임만 봤으니 아직 감지 X
    assert not det.process(kps_up)


# ─── HeadNodDetector ───

def test_head_nod_detected():
    """코 y좌표가 좁은 범위에서 진동 — 끄덕임 (2-3 사이클/1.2초)."""
    det = HeadNodDetector(fps=10.0)
    hits = 0
    for i in range(20):
        # 어깨너비 0.2, nose_y 진동 폭 0.02 (peak-peak), amp_ratio=0.1, 5프레임에 한 사이클
        ny = 0.3 + math.sin(i * math.pi / 2.5) * 0.01
        kps = _kps(nose_y=ny, shoulder_l_x=0.4, shoulder_r_x=0.6)
        if det.process(kps):
            hits += 1
    assert hits >= 1


def test_head_nod_not_detected_when_static():
    det = HeadNodDetector(fps=10.0)
    for _ in range(20):
        kps = _kps(nose_y=0.3)
        assert not det.process(kps)


# ─── HeadShakeDetector ───

def test_head_shake_detected():
    """코 x좌표가 좁은 범위에서 진동 — 도리도리."""
    det = HeadShakeDetector(fps=10.0)
    hits = 0
    for i in range(20):
        # 5프레임에 한 사이클, amp_ratio 0.1
        nx = 0.5 + math.sin(i * math.pi / 2.5) * 0.01
        kps = _kps(nose_x=nx)
        if det.process(kps):
            hits += 1
    assert hits >= 1


def test_head_shake_not_detected_when_static():
    det = HeadShakeDetector(fps=10.0)
    for _ in range(20):
        kps = _kps(nose_x=0.5)
        assert not det.process(kps)


# ─── GazeAtMeDetector ───

def test_gaze_transition_detected():
    """다른 곳 보다가 정면 향하면 감지 (낮은 카메라 — 고개 숙여 로봇 봄)."""
    det = GazeAtMeDetector(fps=10.0, hold_sec=1.2, before_window_sec=2.5)
    # before: nose가 한쪽으로 치우침 (정면 X) — 25프레임
    kps_away = _kps(
        nose_x=0.42, nose_y=0.30, eye_y=0.26, l_eye_x=0.46, r_eye_x=0.54,
    )
    for _ in range(25):
        det.process(kps_away)
    # now: 정면 + 고개 살짝 숙임 (nose가 eye보다 아래) (12프레임)
    kps_facing = _kps(
        nose_x=0.50, nose_y=0.30, eye_y=0.26, l_eye_x=0.46, r_eye_x=0.54,
    )
    hits = sum(1 for _ in range(15) if det.process(kps_facing))
    assert hits == 1


def test_gaze_continuous_facing_not_detected():
    """계속 정면 향하면 (책상 앉아 작업 중) 감지 X."""
    det = GazeAtMeDetector(fps=10.0, hold_sec=1.2, before_window_sec=2.5)
    kps = _kps(
        nose_x=0.50, nose_y=0.30, eye_y=0.26, l_eye_x=0.46, r_eye_x=0.54,
    )
    hits = sum(1 for _ in range(40) if det.process(kps))
    assert hits == 0


def test_gaze_screen_looking_up_not_facing():
    """낮은 카메라 시점에서 사용자가 화면(위)을 보고 있으면 nose가 eye보다
    위 또는 같음 — facing 판정 X. screen 작업 중 false positive 방지."""
    det = GazeAtMeDetector(fps=10.0, hold_sec=1.2, before_window_sec=2.5)
    # x축은 정면이지만 nose가 eye보다 위 (y 작음) — 고개 들어 화면 보는 중
    kps_screen = _kps(
        nose_x=0.50, nose_y=0.24, eye_y=0.26, l_eye_x=0.46, r_eye_x=0.54,
    )
    for _ in range(15):
        det.process(kps_screen)
    # x축 흔들고 다시 정면 화면-바라보기 — pitch 부족하여 facing False
    hits = sum(1 for _ in range(15) if det.process(kps_screen))
    assert hits == 0


def test_gaze_none_clears_state():
    det = GazeAtMeDetector(fps=10.0)
    kps = _kps(nose_x=0.50)
    for _ in range(10):
        det.process(kps)
    det.process(None)
    assert len(det.recent) == 0


# ─── face_orientation ───

def test_orientation_front():
    """양 눈 + 코 모두 보임 → front."""
    kps = _kps(conf=0.9)
    assert face_orientation(kps) == "front"


def test_orientation_side_right():
    """사용자가 자기 오른쪽 봄 → 카메라엔 왼쪽 얼굴만 보임."""
    kps = _kps(conf=0.9)
    # 오른쪽 눈/귀 confidence 0으로 (kps에 r_eye만 conf=0.9, 나머지 그대로)
    kps[KP_R_EYE] = [0.54, 0.26, 0.0]  # invisible
    kps[KP_R_EAR] = [0.0, 0.0, 0.0]
    kps[KP_L_EAR] = [0.40, 0.26, 0.9]   # visible
    kps[KP_NOSE] = [0.5, 0.3, 0.05]     # 코도 잘 안보임 (옆모습)
    assert face_orientation(kps) == "side_right"


def test_orientation_side_left():
    """사용자가 자기 왼쪽 봄 → 카메라엔 오른쪽 얼굴만."""
    kps = _kps(conf=0.9)
    kps[KP_L_EYE] = [0.46, 0.26, 0.0]
    kps[KP_L_EAR] = [0.0, 0.0, 0.0]
    kps[KP_R_EAR] = [0.60, 0.26, 0.9]
    kps[KP_NOSE] = [0.5, 0.3, 0.05]
    assert face_orientation(kps) == "side_left"


def test_orientation_away():
    """얼굴 keypoints 다 안 보이고 어깨만 → away."""
    kps = _kps(conf=0.9)
    kps[KP_NOSE] = [0.0, 0.0, 0.0]
    kps[KP_L_EYE] = [0.0, 0.0, 0.0]
    kps[KP_R_EYE] = [0.0, 0.0, 0.0]
    kps[KP_L_EAR] = [0.0, 0.0, 0.0]
    kps[KP_R_EAR] = [0.0, 0.0, 0.0]
    assert face_orientation(kps) == "away"


def test_orientation_unknown_no_shoulders():
    kps = _kps(conf=0.0)   # 다 invisible
    assert face_orientation(kps) == "unknown"
    assert face_orientation(None) == "unknown"
