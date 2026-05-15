"""인지(Perception) 상태 — 여러 task가 공유하는 가벼운 데이터.

- vision_task가 카메라 결과로 업데이트
- head_tracker가 읽고 서보 제어
- 다른 task도 참조 가능
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class PerceptionState:
    """현재 인지된 환경 상태 — 여러 task가 공유."""

    # 사람 감지
    person_present: bool = False
    person_bbox: tuple[float, float, float, float] | None = None
    person_distance_cm: float = -1.0
    last_person_seen_at: float = 0.0

    # 환경
    temperature_c: float | None = None
    humidity_pct: float | None = None

    def update_person(self, bbox: tuple[float, float, float, float] | None,
                      distance_cm: float = -1.0) -> None:
        if bbox is not None:
            self.person_present = True
            self.person_bbox = bbox
            self.person_distance_cm = distance_cm
            self.last_person_seen_at = time.time()
        # 없을 때는 명시적으로 clear되지 않음 (person_detector가 timeout 관리)

    def clear_person(self) -> None:
        self.person_present = False
        self.person_bbox = None
        self.person_distance_cm = -1.0

    @property
    def person_bbox_center(self) -> tuple[float, float]:
        """bbox 중심점 (0~1 정규화). person_present일 때만 유효."""
        if not self.person_bbox:
            return (0.5, 0.5)
        x0, y0, x1, y1 = self.person_bbox
        return ((x0 + x1) / 2, (y0 + y1) / 2)
