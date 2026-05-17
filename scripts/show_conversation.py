"""대화 로그 보기 — 최근 N분 timeline.

사용:
    python -m scripts.show_conversation             # 최근 30분
    python -m scripts.show_conversation 120         # 최근 2시간
    python -m scripts.show_conversation 60 --user   # user 발화만
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from src.brain import memory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("minutes", nargs="?", type=float, default=30.0)
    parser.add_argument("--user", action="store_true", help="user 발화만")
    parser.add_argument("--robot", action="store_true", help="robot 발화만")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    memory.init_db()
    rows = memory.recent_conversation(minutes=args.minutes, limit=args.limit)
    if args.user:
        rows = [r for r in rows if r["speaker"] == "user"]
    elif args.robot:
        rows = [r for r in rows if r["speaker"] == "robot"]

    if not rows:
        print(f"최근 {args.minutes}분 동안 대화 없음")
        return

    for r in rows:
        ts = datetime.fromtimestamp(r["ts"]).strftime("%H:%M:%S")
        kind = f"[{r['kind']}]" if r["kind"] else ""
        icon = "🗣️" if r["speaker"] == "robot" else "👤"
        print(f"{ts} {icon} {kind} {r['text']}")


if __name__ == "__main__":
    main()
