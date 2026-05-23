"""인지(Perception) 상태 — 여러 task가 공유하는 가벼운 데이터.

- vision_task가 카메라 결과로 업데이트
- head_tracker가 읽고 서보 제어
- 다른 task도 참조 가능
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class PerceptionState:
    """현재 인지된 환경 상태 — 여러 task가 공유."""

    # 사람 감지
    person_present: bool = False
    person_bbox: tuple[float, float, float, float] | None = None
    person_distance_cm: float = -1.0
    last_person_seen_at: float = 0.0
    # pose 모드의 smoothed keypoints (17, 3) — posture_monitor 등이 참조
    last_pose_keypoints: Any = None
    # 가장 최근 카메라 frame (numpy HxWx3 RGB) — agent vision용. None 가능.
    last_frame: Any = None
    last_frame_at: float = 0.0

    # 환경
    temperature_c: float | None = None
    humidity_pct: float | None = None

    # 사용자 표정 (emotion_mirror 최신 결과). neutral/smile/surprised/sad/None.
    current_emotion: str | None = None
    current_emotion_at: float = 0.0

    # 활동 추론 — agent의 더 똑똑한 판단용.
    # gaze_target: front(모니터 응시) / down(책상·핸드폰) / side(딴 곳) / None
    gaze_target: str | None = None
    gaze_target_at: float = 0.0
    # activity_level: still(거의 안 움직임) / focused(집중) / normal / restless(산만)
    activity_level: str | None = None
    activity_level_at: float = 0.0
    # posture_category: upright / slouched / leaning / None
    posture_category: str | None = None
    posture_category_at: float = 0.0

    # 내 머리(카메라)가 향한 각도 — head_tracker가 명령 보낸 직후 갱신.
    # PAN_CENTER_DEG(135) 기준 — 작으면 한쪽, 크면 반대쪽.
    head_pan_deg: float | None = None
    head_tilt_deg: float | None = None
    head_angle_at: float = 0.0

    def update_person(self, bbox: tuple[float, float, float, float] | None,
                      distance_cm: float = -1.0,
                      keypoints: Any = None) -> None:
        if bbox is not None:
            self.person_present = True
            self.person_bbox = bbox
            self.person_distance_cm = distance_cm
            self.last_person_seen_at = time.time()
            if keypoints is not None:
                self.last_pose_keypoints = keypoints

    def clear_person(self) -> None:
        self.person_present = False
        self.person_bbox = None
        self.person_distance_cm = -1.0
        self.last_pose_keypoints = None

    @property
    def person_bbox_center(self) -> tuple[float, float]:
        """bbox 중심점 (0~1 정규화). person_present일 때만 유효."""
        if not self.person_bbox:
            return (0.5, 0.5)
        x0, y0, x1, y1 = self.person_bbox
        return ((x0 + x1) / 2, (y0 + y1) / 2)
