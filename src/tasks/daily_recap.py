"""하루 회고 — 매일 정해진 시각(기본 21:50)에 그날 데이터 요약 발화.

데이터 출처:
- work_sessions: 총 작업 시간
- conversation_log: 대화 turn 수, 자세/제스처 이벤트 횟수
- face_snapshots: 감정 분포 (smile 비율)
- robot_stats: 로봇 컨디션
- env_log: 평균 온도 (있으면)

Claude 가용 시 자연어 한두 문장 생성, 없으면 템플릿.
사용자 존재할 때만 발화. 부재 시 다음 날까지 보류.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta

from src.brain import memory, stats as robot_stats
from src.brain.state_machine import StateContext
from src.face.renderer import FaceState
from src.tasks import behavior_speaker
from src.utils.logger import get_logger

log = get_logger("daily_recap")


RECAP_HOUR = 21
RECAP_MINUTE = 50
CHECK_INTERVAL_SEC = 60.0


def _today_start_ts() -> float:
    return datetime.now().replace(
        hour=0, minute=0, second=0, microsecond=0,
    ).timestamp()


def _gather_today() -> dict:
    today_ts = _today_start_ts()
    data: dict = {}
    # 작업 시간
    try:
        data["work_minutes"] = int(memory.today_total_seconds() / 60)
    except Exception:
        data["work_minutes"] = 0
    # 대화/이벤트 카운트 (conversation_log 기반)
    try:
        from sqlite3 import OperationalError
        with memory.db() as conn:
            rows = conn.execute(
                "SELECT kind, COUNT(*) AS c FROM conversation_log "
                "WHERE ts >= ? GROUP BY kind",
                (today_ts,),
            ).fetchall()
            data["events"] = {r["kind"]: r["c"] for r in rows if r["kind"]}
    except Exception:
        data["events"] = {}
    # 사진 감정 분포
    try:
        with memory.db() as conn:
            rows = conn.execute(
                "SELECT emotion, COUNT(*) AS c FROM face_snapshots "
                "WHERE ts >= ? GROUP BY emotion",
                (today_ts,),
            ).fetchall()
            data["emotions"] = {(r["emotion"] or "unknown"): r["c"] for r in rows}
    except Exception:
        data["emotions"] = {}
    return data


def _fallback_recap(data: dict) -> str:
    parts = []
    wm = data.get("work_minutes", 0)
    if wm > 30:
        parts.append(f"오늘 {wm}분 일했어")
    emo = data.get("emotions", {})
    if emo.get("smile", 0) >= 3:
        parts.append("표정도 자주 웃어줘서 좋았어")
    elif emo.get("smile", 0) > 0:
        parts.append("간간이 웃는 모습 봤어")
    events = data.get("events", {})
    bad = events.get("bad_posture", 0)
    if bad >= 2:
        parts.append(f"자세 안 좋은 거 {bad}번 알려줬는데 신경 좀 써")
    if not parts:
        return "오늘 하루 수고 많았어. 푹 쉬어."
    return ". ".join(parts) + ". 푹 쉬어."


async def run_daily_recap(face: FaceState, ctx: StateContext) -> None:
    log.info(
        f"daily recap 시작 — 매일 {RECAP_HOUR:02d}:{RECAP_MINUTE:02d} 자동 발화"
    )
    last_recap_date: str | None = None

    while True:
        await asyncio.sleep(CHECK_INTERVAL_SEC)
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")

        # 오늘 이미 발화했으면 다음 날까지 skip
        if last_recap_date == today_str:
            continue

        # 발화 시각 도달했고 사용자 존재할 때만
        if now.hour < RECAP_HOUR or (
            now.hour == RECAP_HOUR and now.minute < RECAP_MINUTE
        ):
            continue
        if not ctx.user_present:
            # 시간 지났어도 사람 없으면 대기 (당일 안에 다시 시도)
            # 자정 넘어가면 last_recap_date 안 갱신돼서 자동으로 그 날은 skip됨
            continue

        # 발화
        data = _gather_today()
        log.info(f"recap data: {data}")

        # Claude로 자연 멘트 시도
        msg = ""
        try:
            from src.brain import conversation
            desc = (
                f"하루 회고 시각이 됨 (21시 50분). 오늘 사용자가 한 일 요약:\n"
                f"- 작업 시간 {data.get('work_minutes', 0)}분\n"
                f"- 표정 통계: {data.get('emotions', {})}\n"
                f"- 이벤트 횟수: {data.get('events', {})}\n"
                f"- 내 컨디션: {robot_stats.mood_label()}\n"
                "사용자에게 자연스럽게 회고 한두 문장. 잔소리 X, 따뜻하게."
            )
            msg = conversation.generate_situational(
                desc, max_tokens=120,
                extra=data,
            )
        except Exception as e:
            log.debug(f"recap Claude 실패: {e}")
        if not msg:
            msg = _fallback_recap(data)

        behavior_speaker.say(
            face, ctx, msg,
            kind="daily_recap",
            cooldown_sec=20 * 3600.0,   # 하루 한 번이면 충분
        )
        last_recap_date = today_str
        memory.log_user("(하루 회고 발화)", kind="daily_recap_trigger")
