"""제스처 즉시 응답 멘트 SSOT.

짧은 반사적 멘트(hands_up, nod, shake, gaze)는 여기서 관리.
LLM 거치지 않는 즉각 반응 전용 — agent.py / triggers.py와 분리.

NOTE: wave 응답은 시간대/이름 컨텍스트가 필요해
behavior_speaker.wave_back_message가 SSOT — 여기서 다루지 않음.
"""

from __future__ import annotations

import random

HANDS_UP_REPLIES: tuple[str, ...] = (
    "와! 만세!",
    "야호!",
    "신난다!",
    "오, 뭐가 좋은 일 있어?",
    "축하해!",
)

HEAD_NOD_REPLIES: tuple[str, ...] = (
    "응응.",
    "그래!",
    "오케이!",
    "알겠어.",
    "좋아.",
)

HEAD_SHAKE_REPLIES: tuple[str, ...] = (
    "안돼?",
    "왜?",
    "음... 알겠어.",
    "아냐?",
    "그래, 그러지 말자.",
)

GAZE_REPLIES: tuple[str, ...] = (
    "왜?",
    "응? 무슨 일?",
    "왜 그래?",
    "나 부른 거야?",
    "응, 봐.",
    "뭐 해줄까?",
    "음, 부르려고?",
)


THUMB_UP_REPLIES = (
    "오, 잘됐어?", "굳!", "좋네!", "엄지척이야?", "잘하고 있어.",
)
THUMB_DOWN_REPLIES = (
    "별로야?", "음... 아쉽네.", "왜, 뭐가 잘 안 됐어?", "그래, 다음엔 잘 될 거야.",
)
VICTORY_REPLIES = (
    "오! 뭐 이긴 거야?", "V사인 좋다.", "기분 좋아 보이네.", "축하해!",
)
OPEN_PALM_REPLIES = (
    "오, 손바닥?", "하이파이브?", "왜?", "뭐 줄 거 있어?",
)
FIST_REPLIES = (
    "오, 주먹!", "강하다!", "화이팅이야?", "음, 뭔가 결심한 표정.",
)
POINTING_REPLIES = (
    "뭐 가리킨 거야?", "어? 어디?", "오, 저거?",
)
ILOVEYOU_REPLIES = (
    "오, 사랑해! 나도.", "고마워!", "헤헤 부끄럽다.",
)

GESTURE_POOLS: dict[str, tuple[str, ...]] = {
    "hands_up": HANDS_UP_REPLIES,
    "nod": HEAD_NOD_REPLIES,
    "shake": HEAD_SHAKE_REPLIES,
    "gaze": GAZE_REPLIES,
    "thumb_up": THUMB_UP_REPLIES,
    "thumb_down": THUMB_DOWN_REPLIES,
    "victory": VICTORY_REPLIES,
    "open_palm": OPEN_PALM_REPLIES,
    "fist": FIST_REPLIES,
    "pointing": POINTING_REPLIES,
    "iloveyou": ILOVEYOU_REPLIES,
}


# Claude한테 넘길 자연어 상황 설명 — 짧은 kind보다 의미 명확
GESTURE_DESCRIPTIONS: dict[str, str] = {
    "hands_up": "사용자가 양손을 머리 위로 들어 만세를 함",
    "nod":      "사용자가 고개를 위아래로 끄덕임 (긍정/yes)",
    "shake":    "사용자가 고개를 좌우로 흔듦 (부정/no)",
    "gaze":     "사용자가 로봇을 정면으로 쳐다봄 (시선 전환)",
    "thumb_up":   "사용자가 엄지를 위로 올려 보임 (👍, 잘했다/좋다)",
    "thumb_down": "사용자가 엄지를 아래로 내려 보임 (👎, 별로/안좋아)",
    "victory":    "사용자가 V사인을 보임 (✌️, 승리/기쁨)",
    "open_palm":  "사용자가 손바닥을 펴서 보임 (🖐️, 하이파이브나 멈춤)",
    "fist":       "사용자가 주먹을 보임 (👊, 화이팅/결의)",
    "pointing":   "사용자가 검지로 어딘가를 가리킴 (☝️)",
    "iloveyou":   "사용자가 ILY 손모양을 보임 (🤟, 사랑해)",
}


def pick(kind: str, ctx=None) -> str:
    """제스처 kind에 맞는 멘트. Claude 우선, 실패 시 풀 fallback."""
    pool = GESTURE_POOLS[kind]
    try:
        from src.brain import conversation, memory
        try:
            recent = memory.recent_conversation(minutes=15.0, limit=6)
        except Exception:
            recent = None
        # 사용자 행동 자연어 설명 (Claude 이해 도움)
        desc = GESTURE_DESCRIPTIONS.get(kind, f"제스처 {kind}")
        msg = conversation.generate_situational(
            desc,
            user_name=(getattr(ctx, "user_name", None) if ctx else None),
            recent_dialog=recent,
            max_tokens=60,
        )
        if msg:
            return msg
    except Exception:
        pass
    return random.choice(pool)
