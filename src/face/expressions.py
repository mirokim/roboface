"""표정 프리셋 — Expression dataclass + 기본 12종.

각 표정은 눈/입 파라미터 묶음. State machine이나 brain이 골라서 적용.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EyeShape(Enum):
    NORMAL = "normal"        # 채워진 타원 (기본)
    HAPPY = "happy"          # ⌒⌒ (위로 휘어진 호)
    SQUINT = "squint"        # >_< (찡그린 행복 / 신남)
    SLEEPY = "sleepy"        # ─ (눈 거의 감김)
    SURPRISED = "surprised"  # 큰 타원
    WORRIED = "worried"      # 작은 처진 눈
    ANGRY = "angry"          # > < 바깥쪽 (찡그림)
    LOVE = "love"            # ♥ ♥
    STAR = "star"            # ★ ★ (감탄, 동경)
    DIZZY = "dizzy"          # @ @ (어지러움, 멍함)
    WINK_LEFT = "wink_left"
    WINK_RIGHT = "wink_right"
    CLOSED = "closed"        # 깜빡임 중간 / 절전


class MouthShape(Enum):
    NEUTRAL = "neutral"      # 가벼운 호
    SMILE = "smile"          # 큰 미소
    GRIN = "grin"            # 활짝
    O = "o"                  # 작은 동그란 입 (놀람)
    BIG_O = "big_o"          # 큰 동그란 입 (충격)
    SAD = "sad"              # 처진 입
    FLAT = "flat"            # 무표정 일자
    WAVY = "wavy"            # 물결 입 (어지러움/혼란)
    OPEN_SMALL = "open_small"
    OPEN_MID = "open_mid"
    OPEN_LARGE = "open_large"


@dataclass(frozen=True)
class Expression:
    """표정 프리셋."""

    name: str
    eye: EyeShape
    mouth: MouthShape

    def __repr__(self) -> str:
        return f"Expression({self.name})"


# === 기본 12종 ===
NEUTRAL = Expression("neutral", EyeShape.NORMAL, MouthShape.NEUTRAL)
HAPPY = Expression("happy", EyeShape.HAPPY, MouthShape.SMILE)
EXCITED = Expression("excited", EyeShape.SQUINT, MouthShape.GRIN)
SLEEPY = Expression("sleepy", EyeShape.SLEEPY, MouthShape.NEUTRAL)
SURPRISED = Expression("surprised", EyeShape.SURPRISED, MouthShape.O)
WORRIED = Expression("worried", EyeShape.WORRIED, MouthShape.SAD)
SAD = Expression("sad", EyeShape.WORRIED, MouthShape.SAD)
FOCUSED = Expression("focused", EyeShape.NORMAL, MouthShape.FLAT)
LOVE = Expression("love", EyeShape.LOVE, MouthShape.SMILE)
THINKING = Expression("thinking", EyeShape.NORMAL, MouthShape.FLAT)
WINK = Expression("wink", EyeShape.WINK_LEFT, MouthShape.SMILE)
ANGRY = Expression("angry", EyeShape.ANGRY, MouthShape.SAD)

# === 추가 8종 ===
CONTENT = Expression("content", EyeShape.SLEEPY, MouthShape.SMILE)        # 만족/평온
PROUD = Expression("proud", EyeShape.NORMAL, MouthShape.GRIN)             # 뿌듯/자신
SHOCKED = Expression("shocked", EyeShape.SURPRISED, MouthShape.BIG_O)     # 충격
DIZZY = Expression("dizzy", EyeShape.DIZZY, MouthShape.WAVY)              # 어지러움
STARSTRUCK = Expression("starstruck", EyeShape.STAR, MouthShape.GRIN)     # 감탄/동경
YAWN = Expression("yawn", EyeShape.SLEEPY, MouthShape.O)                  # 하품
CURIOUS = Expression("curious", EyeShape.NORMAL, MouthShape.O)            # 호기심
WINK_R = Expression("wink_r", EyeShape.WINK_RIGHT, MouthShape.SMILE)      # 반대쪽 윙크


EXPRESSIONS_BY_NAME: dict[str, Expression] = {
    e.name: e
    for e in [
        NEUTRAL, HAPPY, EXCITED, SLEEPY, SURPRISED, WORRIED, SAD,
        FOCUSED, LOVE, THINKING, WINK, ANGRY,
        CONTENT, PROUD, SHOCKED, DIZZY, STARSTRUCK, YAWN, CURIOUS, WINK_R,
    ]
}


def get(name: str) -> Expression:
    """이름으로 조회. 없으면 NEUTRAL."""
    return EXPRESSIONS_BY_NAME.get(name, NEUTRAL)
