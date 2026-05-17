"""시간대 분류 SSOT — 아침/점심/오후/저녁/심야.

agent.py / behavior_speaker.py / mood_drift / triggers 등이 공유.
한국어/영문 라벨 둘 다 제공.
"""

from __future__ import annotations

from datetime import datetime


# 경계값(시) — 다른 모듈은 이걸 import해서 쓸 것. 변경은 한 곳에서만.
PERIOD_BOUNDS = {
    "morning": (5, 11),     # 05:00~10:59
    "lunch":   (11, 14),    # 11:00~13:59
    "afternoon": (14, 18),  # 14:00~17:59
    "evening": (18, 22),    # 18:00~21:59
    # late: 22:00~04:59 (그 외 전부)
}

PERIOD_KO = {
    "morning": "아침",
    "lunch":   "점심",
    "afternoon": "오후",
    "evening": "저녁",
    "late":    "심야",
}


def period_for(now: datetime | None = None) -> str:
    """현재 시간대 키 (morning/lunch/afternoon/evening/late)."""
    h = (now or datetime.now()).hour
    for key, (lo, hi) in PERIOD_BOUNDS.items():
        if lo <= h < hi:
            return key
    return "late"


def period_ko(now: datetime | None = None) -> str:
    """현재 시간대 한국어 라벨."""
    return PERIOD_KO[period_for(now)]
