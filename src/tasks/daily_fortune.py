"""오늘의 운세 — 매일 정해진 시각(기본 11:00)에 가벼운 운세 한두 문장.

운세는 진지한 점술이 아니라 *가벼운 동반자 톤*의 한마디. 잔소리 X.

발화 조건:
- 시각 도달 (기본 11:00)
- 사용자 존재
- 오늘 아직 발화 안 함
- quiet hours 무관 (사용자가 명시적으로 원한 시간)

Claude 가용 시 자연어 생성, 없으면 풀 fallback.
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime

from src.brain import memory, stats as robot_stats
from src.brain.state_machine import StateContext
from src.face.expressions import HAPPY
from src.face.renderer import FaceState
from src.tasks import behavior_speaker
from src.utils.logger import get_logger

log = get_logger("daily_fortune")


FORTUNE_HOUR = 11
FORTUNE_MINUTE = 0
CHECK_INTERVAL_SEC = 60.0
# 시각 지나도 사용자 부재면 +N분 안엔 발화 시도. 그 이후엔 그날 skip.
GRACE_MINUTES = 180   # 11:00 ~ 14:00 사이에 사용자 들어오면 발화


_FALLBACK_FORTUNES = (
    "오늘은 작은 우연이 좋은 쪽으로 흐를 거야.",
    "음, 오늘 만나는 사람한테 살짝 더 부드럽게 굴면 좋겠어.",
    "흠, 오늘은 너무 멀리 보지 말고 가까운 거 하나만 잘하면 돼.",
    "오늘은 의외의 한마디가 기억에 남을 거야.",
    "음, 점심 메뉴를 평소랑 다른 걸로 골라봐. 그게 운이야.",
    "흠, 오늘은 잠깐 멈춰서 하늘 한 번 보면 좋은 기운 와.",
    "오늘 좀 천천히 가도 돼. 서두를수록 꼬여.",
    "음, 오늘은 누구 한 사람한테 짧게 안부 한마디 — 그게 돌아올 거야.",
    "흠, 오늘은 한 일보다 *안 한 일*이 더 큰 결정이야.",
    "오늘은 작은 정리 하나가 큰 흐름 바꿔. 책상 한 켠이라도.",
    "음, 오늘 받는 메시지 하나 잘 읽어봐. 거기 답이 있어.",
    "흠, 오늘은 새로운 시도보다 익숙한 거 깊게 들어가는 날.",
)


def _fallback_fortune(ctx: StateContext) -> str:
    base = random.choice(_FALLBACK_FORTUNES)
    name = getattr(ctx, "user_name", None)
    if name and random.random() < 0.4:
        return f"{name}아, {base}"
    return base


async def run_daily_fortune(face: FaceState, ctx: StateContext) -> None:
    log.info(
        f"daily fortune 시작 — 매일 {FORTUNE_HOUR:02d}:{FORTUNE_MINUTE:02d} "
        f"(부재 시 +{GRACE_MINUTES}분 grace) 자동 발화"
    )
    last_fortune_date: str | None = None

    while True:
        await asyncio.sleep(CHECK_INTERVAL_SEC)
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")

        if last_fortune_date == today_str:
            continue

        # 발화 시각 전이면 대기
        target_min = FORTUNE_HOUR * 60 + FORTUNE_MINUTE
        cur_min = now.hour * 60 + now.minute
        if cur_min < target_min:
            continue
        # grace 지나면 그날 skip (자정 넘으면 자동으로 재시도)
        if cur_min > target_min + GRACE_MINUTES:
            continue
        if not ctx.user_present:
            # 시간 지났어도 사람 없으면 grace 안에서 재시도
            continue

        # Claude로 자연 운세 시도
        msg = ""
        weekday_ko = ["월", "화", "수", "목", "금", "토", "일"][now.weekday()]
        try:
            from src.brain import conversation
            desc = (
                f"오늘의 운세 시각 — {now.strftime('%Y-%m-%d')} ({weekday_ko}요일).\n"
                f"진지한 점술 X, 가벼운 동반자 톤의 한두 문장 운세를 자연스럽게.\n"
                f"내 컨디션: {robot_stats.mood_label()}\n"
                "잔소리/충고 X. 살짝 시적이거나 농담 섞은 한마디.\n"
                "예시 톤: '음, 오늘은 의외의 한마디가 기억에 남을 거야', "
                "'흠, 점심 메뉴 평소랑 다른 거로'.\n"
                "이모지 X, 괄호 무대지문 X. 순수 발화만."
            )
            msg = conversation.generate_situational(
                desc, max_tokens=120,
                extra={"weekday": weekday_ko, "hour": now.hour},
            )
        except Exception as e:
            log.debug(f"fortune Claude 실패: {e}")
        if not msg:
            msg = _fallback_fortune(ctx)

        log.info(f"🔮 오늘의 운세: {msg}")
        behavior_speaker.say(
            face, ctx, msg,
            kind="daily_fortune",
            cooldown_sec=20 * 3600.0,   # 하루 한 번
            expression=HAPPY,
            bypass_quiet=True,           # 사용자가 명시적으로 원한 시간
        )
        last_fortune_date = today_str
        memory.log_user("(오늘의 운세 발화)", kind="daily_fortune_trigger")
