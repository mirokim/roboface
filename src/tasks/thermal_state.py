"""온도에 따라 얼굴 표정에 sweat / shiver 반영.

DHT22 센서 결과 (perception.temperature_c) → face.sweat_intensity / shiver_intensity.
부드럽게 보간 (튀지 않게).

기준 (°C):
- ≥30: 땀 강하게 (1.0)
- 28~30: 점진적 (0.0~1.0)
- 18~28: 평온 (0.0)
- 15~18: 점진적 떨림
- ≤15: 떨림 강하게 (1.0)
"""

from __future__ import annotations

import asyncio

from src.brain.perception import PerceptionState
from src.face.renderer import FaceState
from src.utils.logger import get_logger

log = get_logger("thermal_state")

UPDATE_INTERVAL_SEC = 2.0
BLEND = 0.2   # 매 호출에 target과의 차이를 이만큼 보정 — 약 10초에 도달

HOT_START_C = 28.0
HOT_FULL_C = 30.0
COLD_FULL_C = 15.0
COLD_START_C = 18.0


def _targets_for(temp_c: float | None) -> tuple[float, float]:
    if temp_c is None:
        return 0.0, 0.0
    # 더위
    if temp_c >= HOT_FULL_C:
        sweat = 1.0
    elif temp_c >= HOT_START_C:
        sweat = (temp_c - HOT_START_C) / (HOT_FULL_C - HOT_START_C)
    else:
        sweat = 0.0
    # 추위
    if temp_c <= COLD_FULL_C:
        shiver = 1.0
    elif temp_c <= COLD_START_C:
        shiver = (COLD_START_C - temp_c) / (COLD_START_C - COLD_FULL_C)
    else:
        shiver = 0.0
    return sweat, shiver


async def run_thermal_state(face: FaceState, perception: PerceptionState) -> None:
    log.info("thermal_state 시작")
    while True:
        await asyncio.sleep(UPDATE_INTERVAL_SEC)
        target_sweat, target_shiver = _targets_for(perception.temperature_c)
        face.sweat_intensity += (target_sweat - face.sweat_intensity) * BLEND
        face.shiver_intensity += (target_shiver - face.shiver_intensity) * BLEND
        # raw 환경 값도 face에 미러 — 우하단 오버레이용
        face.env_temp_c = perception.temperature_c
        face.env_humidity_pct = perception.humidity_pct
