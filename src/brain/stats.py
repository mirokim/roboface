"""Tamagotchi 스타일 4개 스탯 — 시간 따라 decay, 이벤트로 회복.

energy   — 0~100. 활동 중 감소, quiet hours에 회복. <30이면 졸린 톤.
mood     — 0~100. 시간 따라 감소, 긍정 상호작용으로 회복. <30이면 우울.
social   — 0~100. 혼자 있으면 감소, 사용자 있으면 회복. <30이면 외로움.
curiosity — 0~100. 평소 천천히 감소, 새 이벤트(제스처/얼굴/대화)로 회복. <30이면 심심.

스탯은 표정 baseline + Claude 멘트 톤에 영향. agent._build_situation에 자동 포함.
DB: user_patterns 키 "robot_stats" (JSON). 마지막 업데이트 시각도 같이 저장.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime

from src.brain import memory
from src.brain.time_of_day import period_for
from src.utils.logger import get_logger

log = get_logger("stats")


# 시간당 decay (스탯 -X/hour)
DECAY_PER_HOUR = {
    "energy":   2.5,    # 평소 활동 — 40시간이면 0
    "mood":     1.5,
    "social":   6.0,    # 혼자면 빨리 줄어듦
    "curiosity": 3.5,
}

# 회복 (per hour 또는 per event)
ENERGY_SLEEP_RECOVER = 12.0    # quiet hours 동안 시간당
SOCIAL_PRESENCE_RECOVER = 8.0  # 사용자 있을 때 시간당
MOOD_PRESENCE_RECOVER = 1.5    # 사용자 옆에 있는 것만으로도 살짝

# 이벤트 → (스탯, 즉시 변화) 리스트
EVENT_DELTAS: dict[str, list[tuple[str, float]]] = {
    "presence_new":   [("social", 5), ("mood", 2)],
    "wave":           [("social", 4), ("mood", 6)],
    "hands_up":       [("mood", 10), ("curiosity", 4)],
    "nod":            [("social", 1), ("mood", 1)],
    "gaze":           [("social", 2)],
    "thumb_up":       [("mood", 8)],
    "thumb_down":     [("mood", -3)],
    "victory":        [("mood", 6), ("curiosity", 2)],
    "iloveyou":       [("mood", 15), ("social", 5)],
    "face_recognize": [("social", 4), ("curiosity", 3)],
    "voice_chat":     [("social", 6), ("mood", 3), ("curiosity", 2)],
    "bad_posture":    [("mood", -1)],
    "chitchat":       [("mood", 1)],
}


@dataclass
class RobotStats:
    energy: float = 75.0
    mood: float = 65.0
    social: float = 50.0
    curiosity: float = 60.0
    updated_at: float = 0.0

    def clamp(self) -> None:
        self.energy = max(0.0, min(100.0, self.energy))
        self.mood = max(0.0, min(100.0, self.mood))
        self.social = max(0.0, min(100.0, self.social))
        self.curiosity = max(0.0, min(100.0, self.curiosity))


_STATS: RobotStats | None = None


def _load() -> RobotStats:
    data = memory.get_pattern("robot_stats")
    if isinstance(data, dict):
        try:
            return RobotStats(**data)
        except TypeError:
            pass
    return RobotStats(updated_at=time.time())


def _save(stats: RobotStats) -> None:
    stats.updated_at = time.time()
    memory.set_pattern("robot_stats", asdict(stats))


def get() -> RobotStats:
    """현재 스탯. 처음 호출 시 lazy load, decay 자동 적용."""
    global _STATS
    if _STATS is None:
        _STATS = _load()
        if _STATS.updated_at == 0:
            _STATS.updated_at = time.time()
            _save(_STATS)
    _tick(_STATS)
    return _STATS


def _tick(stats: RobotStats) -> None:
    """경과 시간만큼 decay/회복 적용."""
    now = time.time()
    elapsed_h = (now - stats.updated_at) / 3600.0
    if elapsed_h <= 0:
        return

    # decay 모두 적용
    for stat, rate in DECAY_PER_HOUR.items():
        cur = getattr(stats, stat)
        setattr(stats, stat, cur - rate * elapsed_h)

    # quiet hours = energy 회복
    if _is_quiet_now():
        stats.energy += ENERGY_SLEEP_RECOVER * elapsed_h

    # 사용자가 옆에 있는지는 caller가 별도로 on_event로 보고. 여기선 decay만.
    stats.clamp()
    stats.updated_at = now
    _save(stats)


def _is_quiet_now() -> bool:
    """time_of_day 'late' 시간대면 수면 시간."""
    return period_for() == "late"


def on_event(kind: str, *, multiplier: float = 1.0) -> None:
    """이벤트 발생 시 스탯 회복/감소. EVENT_DELTAS 매핑."""
    deltas = EVENT_DELTAS.get(kind)
    if not deltas:
        return
    stats = get()
    for stat, delta in deltas:
        setattr(stats, stat, getattr(stats, stat) + delta * multiplier)
    stats.clamp()
    _save(stats)
    log.debug(
        f"stat event '{kind}': "
        f"E={stats.energy:.0f} M={stats.mood:.0f} "
        f"S={stats.social:.0f} C={stats.curiosity:.0f}"
    )


def on_presence_tick(elapsed_sec: float) -> None:
    """사용자가 옆에 있는 동안 주기 호출 — social/mood 천천히 회복."""
    if elapsed_sec <= 0:
        return
    h = elapsed_sec / 3600.0
    stats = get()
    stats.social += SOCIAL_PRESENCE_RECOVER * h
    stats.mood += MOOD_PRESENCE_RECOVER * h
    stats.clamp()
    _save(stats)


# === 표정/멘트 영향 헬퍼 ===

def summary_text() -> str:
    """현재 스탯을 한 줄로 (agent 프롬프트용)."""
    s = get()
    return (
        f"내 상태: 에너지 {s.energy:.0f}/100, 기분 {s.mood:.0f}, "
        f"사회성 {s.social:.0f}, 호기심 {s.curiosity:.0f}"
    )


def mood_label() -> str:
    """전체적인 컨디션 한국어 한 단어."""
    s = get()
    if s.energy < 25:
        return "졸림"
    if s.mood < 25:
        return "우울"
    if s.social < 25:
        return "외로움"
    if s.curiosity < 25:
        return "지루함"
    avg = (s.energy + s.mood + s.social + s.curiosity) / 4
    if avg > 75:
        return "활기참"
    if avg > 55:
        return "괜찮음"
    return "보통"


def suggested_expression() -> str | None:
    """스탯 기반으로 표정 기본값 추천. 평상시 None.

    값이 낮은 스탯이 있으면 그 표정 우선.
    """
    s = get()
    if s.energy < 25:
        return "SLEEPY"
    if s.mood < 25:
        return "SAD"
    if s.social < 25:
        return "SAD"   # 외로움도 SAD 표정으로
    if s.curiosity < 25 and s.energy > 40:
        return "YAWN"
    if s.mood > 80 and s.energy > 70:
        return "HAPPY"
    return None
