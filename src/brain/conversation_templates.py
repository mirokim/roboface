"""제스처 즉시 응답 멘트 SSOT.

짧은 반사적 멘트(hands_up, nod, shake, gaze)는 여기서 관리.
LLM 거치지 않는 즉각 반응 전용 — agent.py / triggers.py와 분리.

NOTE: wave 응답은 시간대/이름 컨텍스트가 필요해
behavior_speaker.wave_back_message가 SSOT — 여기서 다루지 않음.
"""

from __future__ import annotations

import random

HANDS_UP_REPLIES: tuple[str, ...] = (
    "와 만세!", "야호!", "오 신난다!", "뭐 좋은 일 있어?",
    "축하해!", "어 만세네?", "오케이 같이 신나자.",
    "음, 기분 좋은가 봐.", "오 하이!", "와 진짜?",
    "흠, 무슨 일?",
)

HEAD_NOD_REPLIES: tuple[str, ...] = (
    "응응.", "그래.", "오케이.", "알겠어.", "좋아.",
    "어 그래.", "음 그래.", "그래그래.", "알았어.",
    "오케이오케이.",
)

HEAD_SHAKE_REPLIES: tuple[str, ...] = (
    "음 아냐?", "왜?", "그래 알겠어.", "어 아니구나.",
    "음 알았어.", "그러지 말자.", "어 별로야?",
    "흠 아냐.", "그래, 안 해.",
)

GAZE_REPLIES: tuple[str, ...] = (
    "응?", "어 왜?", "음, 부른 거야?", "왜 그래?",
    "어 봤어.", "뭐 해줄까?", "응 보고 있어.",
    "어 나야?", "흠?", "음 무슨 일?",
)


THUMB_UP_REPLIES = (
    "오 굳!", "잘했어?", "좋네!", "엄지척이네.",
    "음 잘됐나 봐.", "오케이 좋아.",
)
THUMB_DOWN_REPLIES = (
    "음 별로야?", "어 아쉽네.", "왜 뭐가 안 됐어?",
    "그래 다음엔 잘 될 거야.", "흠, 아쉽다.",
)
VICTORY_REPLIES = (
    "오 이긴 거야?", "V 좋다.", "기분 좋아 보이네.",
    "축하해!", "음, 뭔가 잘 풀렸나 봐.",
)
OPEN_PALM_REPLIES = (
    "오 손바닥?", "하이파이브?", "어 왜?", "뭐 줄 거 있어?",
    "음 멈춰?",
)
FIST_REPLIES = (
    "오 주먹!", "화이팅?", "강하다.", "음 결심한 표정이네.",
    "어 파이팅이야?",
)
POINTING_REPLIES = (
    "뭐 가리킨 거야?", "어 어디?", "오 저거?",
    "음 그쪽?",
)
ILOVEYOU_REPLIES = (
    "오 사랑해 나도.", "고마워.", "헤헤 부끄럽네.",
    "음 감동이야.",
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
        from src.utils.logger import get_logger
        log = get_logger("templates")
        try:
            recent = memory.recent_conversation(minutes=15.0, limit=6)
        except Exception:
            recent = None
        desc = GESTURE_DESCRIPTIONS.get(kind, f"제스처 {kind}")
        msg = conversation.generate_situational(
            desc,
            user_name=(getattr(ctx, "user_name", None) if ctx else None),
            recent_dialog=recent,
            max_tokens=60,
        )
        if msg:
            return msg
        log.debug(f"gesture[{kind}]: Claude empty → 풀 fallback")
    except Exception as e:
        from src.utils.logger import get_logger
        get_logger("templates").debug(f"gesture[{kind}]: Claude 예외 → 풀: {e}")
    return random.choice(pool)
