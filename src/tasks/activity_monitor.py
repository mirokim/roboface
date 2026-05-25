"""활동성(activity_level) 추론 — 사용자가 얼마나 움직이는지 1분 윈도우로 측정.

10초마다 perception.last_pose_keypoints의 nose + shoulder 중심을 샘플링.
6개 샘플(60초) 모이면 좌표 표준편차로 변동량 계산 → 4단계 라벨.

라벨:
  - still     : 거의 안 움직임. 멍 때리거나 깊이 집중. 자세 변화 없음.
  - focused   : 잔잔히 움직임. 키보드 타이핑 정도.
  - normal    : 보통 — 가끔 자세 바꾸거나 손 움직임.
  - restless  : 자주 크게 움직임. 산만, 통화, 자리 들썩임.

agent가 이 라벨을 멘트 톤에 반영 가능 (예: still → 조용히, restless → 살짝 진정 권유).
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any

from src.brain.perception import PerceptionState
from src.utils.logger import get_logger

log = get_logger("activity_monitor")


SAMPLE_INTERVAL_SEC = 10.0
WINDOW_SAMPLES = 6   # 60초 윈도우
KP_CONF_THRESHOLD = 0.20


# 변동량 임계 (위치 정규화 좌표 0~1 기준의 std 평균)
_THRESHOLD_STILL = 0.004
_THRESHOLD_FOCUSED = 0.012
_THRESHOLD_NORMAL = 0.035


def _classify(std_mean: float) -> str:
    if std_mean < _THRESHOLD_STILL:
        return "still"
    if std_mean < _THRESHOLD_FOCUSED:
        return "focused"
    if std_mean < _THRESHOLD_NORMAL:
        return "normal"
    return "restless"


def _sample(keypoints: Any) -> tuple[float, float, float, float] | None:
    """keypoints에서 nose + 양 어깨 중심점만 추출. 신뢰도 부족 시 None."""
    if keypoints is None:
        return None
    nose = keypoints[0]
    l_sh = keypoints[5]
    r_sh = keypoints[6]
    if nose[2] < KP_CONF_THRESHOLD:
        return None
    if l_sh[2] < KP_CONF_THRESHOLD or r_sh[2] < KP_CONF_THRESHOLD:
        return None
    sh_x = float((l_sh[0] + r_sh[0]) / 2)
    sh_y = float((l_sh[1] + r_sh[1]) / 2)
    return (float(nose[0]), float(nose[1]), sh_x, sh_y)


def _compute_std_mean(samples: list[tuple[float, float, float, float]]) -> float:
    """샘플 리스트의 각 차원 표준편차 평균."""
    if len(samples) < 2:
        return 0.0
    n = len(samples)
    means = [sum(s[i] for s in samples) / n for i in range(4)]
    variances = [
        sum((s[i] - means[i]) ** 2 for s in samples) / n
        for i in range(4)
    ]
    stds = [v ** 0.5 for v in variances]
    return sum(stds) / 4


async def run_activity_monitor(perception: PerceptionState) -> None:
    """매 10초 샘플, 60초 윈도우로 activity_level 갱신."""
    window: deque[tuple[float, float, float, float]] = deque(maxlen=WINDOW_SAMPLES)
    log.info("activity monitor 시작")
    last_label: str | None = None
    miss_count = 0
    while True:
        await asyncio.sleep(SAMPLE_INTERVAL_SEC)
        sample = _sample(perception.last_pose_keypoints)
        if sample is None:
            # 사람 없거나 keypoint 부족. 연속 2회(20초) miss 시 윈도우 reset:
            # 옛 위치 + 새 위치(복귀 후) 섞이면 std 폭주해 'restless' 잘못 인식.
            miss_count += 1
            if miss_count >= 2 and window:
                window.clear()
                last_label = None
                perception.activity_level = None
            continue
        miss_count = 0
        window.append(sample)
        if len(window) < WINDOW_SAMPLES:
            continue
        std_mean = _compute_std_mean(list(window))
        label = _classify(std_mean)
        if label != last_label:
            log.debug(f"activity → {label} (std_mean={std_mean:.4f})")
            last_label = label
        perception.activity_level = label
        perception.activity_level_at = time.time()
