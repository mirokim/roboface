"""행동 기억 — 사용자의 하루 행동을 요약해 기억하고, 그걸 근거로 한마디 만든다.

원천은 이미 쌓이는 SQLite 로그(conversation_log의 user 이벤트, work_sessions).
매일 요약(첫 등장/마지막 본 시각/앉은 시간/손인사·끄덕·발화 횟수)을
user_patterns["behavior_day_<YYYY-MM-DD>"]에 저장해 로그가 지워져도(90일) 남는다.

remarks()가 오늘 vs 어제/평소를 비교한 멘트 후보를 돌려주고, 잡담/인사 풀이 섞어 쓴다.
"""

from __future__ import annotations

import statistics
import time
from datetime import date, datetime, timedelta
from typing import Any

from src.brain import memory
from src.utils.logger import get_logger

log = get_logger("behavior_memory")

_KEY = "behavior_day_{}"

# 하루 요약에 세는 user 이벤트 kind → 요약 필드
_COUNT_KINDS = {
    "gesture_wave": "waves",
    "gesture_nod": "nods",
    "gesture_shake": "shakes",
    "gesture_hands_up": "hands_up",
    "ambient": "speech",
    "voice": "speech",
    "gaze_at_me": "gazes",
}


def _day_bounds(d: date) -> tuple[float, float]:
    start = datetime(d.year, d.month, d.day).timestamp()
    return start, start + 86400


def summarize_day(d: date | None = None) -> dict[str, Any]:
    """해당 날짜 행동 요약 계산 (로그에서). 저장은 안 함."""
    d = d or date.today()
    t0, t1 = _day_bounds(d)
    out: dict[str, Any] = {
        "date": d.isoformat(), "arrival": None, "last_seen": None,
        "work_minutes": 0, "waves": 0, "nods": 0, "shakes": 0,
        "hands_up": 0, "speech": 0, "gazes": 0, "robot_said": 0,
    }
    with memory.db() as conn:
        rows = conn.execute(
            "SELECT ts, speaker, kind FROM conversation_log "
            "WHERE ts >= ? AND ts < ? ORDER BY ts",
            (t0, t1),
        ).fetchall()
        for r in rows:
            ts, speaker, kind = float(r["ts"]), r["speaker"], r["kind"] or ""
            if speaker == "user":
                if out["arrival"] is None and kind in ("presence_new", "face_recognized"):
                    out["arrival"] = ts
                out["last_seen"] = ts
                field = _COUNT_KINDS.get(kind)
                if field:
                    out[field] += 1
            elif speaker == "robot" and kind != "cli_speak":
                out["robot_said"] += 1
        # 안 닫힌 세션(재시작/정전)은 다음 세션 시작까지로 잘라서 겹침 합산 방지
        sess = conn.execute(
            "SELECT start_ts, end_ts, duration_sec FROM work_sessions "
            "WHERE start_ts >= ? AND start_ts < ? ORDER BY start_ts",
            (t0, t1),
        ).fetchall()
        total = 0.0
        for i, r in enumerate(sess):
            start = float(r["start_ts"])
            if r["end_ts"] is not None or r["duration_sec"] is not None:
                total += float(r["duration_sec"] or (float(r["end_ts"]) - start))
                continue
            nxt = float(sess[i + 1]["start_ts"]) if i + 1 < len(sess) else min(time.time(), t1)
            total += max(0.0, nxt - start)
        out["work_minutes"] = int(total / 60)
    return out


def save_day(d: date | None = None) -> dict[str, Any]:
    s = summarize_day(d)
    if s["arrival"] is not None or s["work_minutes"] > 0:
        memory.set_pattern(_KEY.format(s["date"]), s)
    return s


def load_day(d: date) -> dict[str, Any] | None:
    return memory.get_pattern(_KEY.format(d.isoformat()))


def recent_days(n: int = 14, include_today: bool = True) -> list[dict[str, Any]]:
    """최근 n일 요약 (저장본 우선, 오늘은 실시간 계산). 오래된 순."""
    out = []
    today = date.today()
    for i in range(n - 1, -1, -1):
        d = today - timedelta(days=i)
        if d == today:
            if include_today:
                out.append(summarize_day(d))
            continue
        s = load_day(d)
        if s is None:
            s = summarize_day(d)
            if s["arrival"] is not None or s["work_minutes"] > 0:
                memory.set_pattern(_KEY.format(s["date"]), s)
        if s.get("arrival") is not None or s.get("work_minutes", 0) > 0:
            out.append(s)
    return out


def _hm(ts: float | None) -> str:
    if not ts:
        return "?"
    t = datetime.fromtimestamp(ts)
    return f"{t.hour}시 {t.minute}분" if t.minute else f"{t.hour}시"


def _minutes_of_day(ts: float) -> int:
    t = datetime.fromtimestamp(ts)
    return t.hour * 60 + t.minute


def typical_arrival_minutes(days: int = 14) -> int | None:
    """최근 N일 등장 시각 중앙값 (분 단위). 데이터 3일 미만이면 None."""
    arr = [
        _minutes_of_day(s["arrival"]) for s in recent_days(days, include_today=False)
        if s.get("arrival")
    ]
    if len(arr) < 3:
        return None
    return int(statistics.median(arr))


def streak_days() -> int:
    """오늘 포함 연속으로 본 날 수."""
    n = 0
    d = date.today()
    while True:
        s = summarize_day(d) if d == date.today() else load_day(d)
        if not s or s.get("arrival") is None:
            break
        n += 1
        d -= timedelta(days=1)
        if n > 60:
            break
    return n


def remarks(now: float | None = None) -> list[str]:
    """지금 상황에서 할 만한 '기억 기반' 한마디 후보들 (없으면 빈 리스트)."""
    now = now or time.time()
    out: list[str] = []
    try:
        today = summarize_day(date.today())
        yday = load_day(date.today() - timedelta(days=1)) or summarize_day(
            date.today() - timedelta(days=1)
        )
    except Exception as e:
        log.debug(f"behavior remarks 실패: {e}")
        return out

    # 등장 시각 비교 (오늘 등장 후 2시간 안에만 의미 있음)
    if today.get("arrival") and now - today["arrival"] < 7200:
        ta = _minutes_of_day(today["arrival"])
        if yday and yday.get("arrival"):
            ya = _minutes_of_day(yday["arrival"])
            diff = ta - ya
            if diff <= -30:
                out.append(f"어제는 {_hm(yday['arrival'])}에 왔는데 오늘은 일찍 왔네.")
            elif diff >= 30:
                out.append(f"어제는 {_hm(yday['arrival'])}에 왔는데 오늘은 좀 늦었네.")
            else:
                out.append("어제랑 비슷한 시간에 왔네. 규칙적이다.")
        typ = typical_arrival_minutes()
        if typ is not None:
            if ta < typ - 40:
                out.append("평소보다 꽤 일찍 왔네.")
            elif ta > typ + 40:
                out.append("평소보다 늦게 왔네. 무슨 일 있었어?")

    # 어제 앉은 시간
    if yday and yday.get("work_minutes", 0) >= 60:
        wm = yday["work_minutes"]
        h, m = divmod(wm, 60)
        dur = f"{h}시간 {m}분" if h else f"{m}분"
        if wm >= 300:
            out.append(f"어제 {dur}이나 앉아있었어. 오늘은 좀 쉬엄쉬엄 해.")
        else:
            out.append(f"어제는 {dur} 정도 같이 있었네.")

    # 연속 출석
    st = streak_days()
    if st >= 3:
        out.append(f"{st}일 연속으로 보네. 좋다.")

    # 이번 주 손인사
    week = recent_days(7)
    waves = sum(s.get("waves", 0) for s in week)
    if waves >= 3 and today.get("waves", 0) > 0:
        out.append(f"이번 주 손인사 {waves}번째야.")

    # 어제 많이 말 걸었는지
    if yday and yday.get("speech", 0) >= 5 and today.get("speech", 0) == 0:
        out.append("어제는 말 많이 걸어줬는데 오늘은 조용하네.")

    return out


def history_text(days: int = 7) -> str:
    """CLI/원격용 요약 텍스트."""
    lines = []
    for s in recent_days(days):
        lines.append(
            f"{s['date']}  등장 {_hm(s.get('arrival'))}  마지막 {_hm(s.get('last_seen'))}  "
            f"앉음 {s.get('work_minutes', 0)}분  손인사 {s.get('waves', 0)}  "
            f"끄덕 {s.get('nods', 0)}  말 {s.get('speech', 0)}  로봇발화 {s.get('robot_said', 0)}"
        )
    return "\n".join(lines) if lines else "(기록 없음)"
