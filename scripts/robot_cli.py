"""roboface 외부 제어 CLI (의존성 없음).

실행 중인 roboface 프로세스에 SQLite 큐로 명령 전송.
명령은 메인 프로세스의 command_executor가 1초 안에 처리.

표준 라이브러리만 사용 — venv 없이 시스템 python으로도 동작.

사용:
    python scripts/robot_cli.py snapshot
    python scripts/robot_cli.py gesture nod
    python scripts/robot_cli.py speak "안녕"
    python scripts/robot_cli.py status
    python scripts/robot_cli.py expressions
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path


def _find_db_path() -> Path:
    """src/config.py를 import 안 하고도 DB_PATH 찾기."""
    # config.py 와 동일 규칙: <repo>/src/data/roboface.db
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "src" / "data" / "roboface.db"


DB_PATH = _find_db_path()


def _connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        print(f"DB 없음: {DB_PATH} — roboface가 한 번이라도 떴는지 확인", file=sys.stderr)
        sys.exit(2)
    conn = sqlite3.connect(str(DB_PATH), timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def enqueue(cmd: str, args: dict) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO command_queue(enqueued_at, cmd, args, status) "
            "VALUES (?, ?, ?, 'pending')",
            (time.time(), cmd, json.dumps(args, ensure_ascii=False)),
        )
        return cur.lastrowid or 0


def fetch_status(cmd_id: int) -> dict | None:
    with _connect() as conn:
        r = conn.execute(
            "SELECT id, cmd, status, result FROM command_queue WHERE id=?",
            (cmd_id,),
        ).fetchone()
    return dict(r) if r else None


def wait_for(cmd_id: int, timeout: float = 10.0) -> dict | None:
    end = time.time() + timeout
    while time.time() < end:
        st = fetch_status(cmd_id)
        if st and st["status"] in ("done", "failed"):
            return st
        time.sleep(0.2)
    return None


# 표정 목록 — expressions.py와 동기화. 직접 import 안 하고 정적 리스트로 유지.
EXPRESSION_NAMES = sorted([
    "neutral", "happy", "excited", "sleepy", "surprised", "worried", "sad",
    "focused", "love", "thinking", "wink", "angry",
    "content", "proud", "shocked", "dizzy", "starstruck", "yawn", "curious",
    "wink_r", "confused", "annoyed",
])

STATE_NAMES = ["IDLE", "WATCHING", "GREETING", "TALKING", "ALERTING", "LISTENING"]
GESTURE_KINDS = ["wave", "hands_up", "nod", "shake", "gaze",
                 "presence_new", "presence_left"]
POSE_KINDS = ["nod", "shake", "greeting", "tilt_curious", "look_around"]


def submit_and_report(cmd: str, args: dict, wait: bool) -> int:
    cmd_id = enqueue(cmd, args)
    print(f"# cmd #{cmd_id}: {cmd} {args}")
    if not wait:
        return 0
    st = wait_for(cmd_id)
    if st is None:
        print("(timeout - 메인 프로세스가 안 떠있나?)")
        return 2
    print(f"status: {st['status']}")
    if st.get("result"):
        print(f"result: {st['result']}")
    return 0 if st["status"] == "done" else 1


def main() -> int:
    p = argparse.ArgumentParser(prog="robot_cli")
    p.add_argument("--wait", action="store_true", help="처리 완료까지 대기")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("speak"); sp.add_argument("text")
    sp.add_argument("--expression", "-e", default=None)

    se = sub.add_parser("expression"); se.add_argument("name")

    sd = sub.add_parser("dance")
    sd.add_argument("--beats", type=int, default=4)
    sd.add_argument("--bpm", type=int, default=120)

    sp2 = sub.add_parser("pose"); sp2.add_argument("kind", choices=POSE_KINDS)

    st = sub.add_parser("transition"); st.add_argument("state", choices=STATE_NAMES)

    sg = sub.add_parser("gesture", help="vision 우회해 sensor event emit")
    sg.add_argument("kind", choices=GESTURE_KINDS)

    sub.add_parser("blink")
    sub.add_parser("status", help="현재 상태 (--wait 자동)")
    sub.add_parser("expressions", help="사용 가능한 표정 목록")

    ss = sub.add_parser(
        "snapshot",
        help="카메라 프레임 + bbox/keypoint → /tmp/roboface_debug.jpg",
    )
    ss.add_argument("--note", default="")

    args = p.parse_args()

    if args.cmd == "expressions":
        for n in EXPRESSION_NAMES:
            print(n)
        return 0

    wait = args.wait or args.cmd == "status"

    if args.cmd == "speak":
        return submit_and_report(
            "speak", {"text": args.text, "expression": args.expression}, wait,
        )
    if args.cmd == "expression":
        return submit_and_report("expression", {"name": args.name}, wait)
    if args.cmd == "dance":
        return submit_and_report(
            "dance", {"beats": args.beats, "bpm": args.bpm}, wait,
        )
    if args.cmd == "pose":
        return submit_and_report("pose", {"kind": args.kind}, wait)
    if args.cmd == "transition":
        return submit_and_report("transition", {"state": args.state}, wait)
    if args.cmd == "gesture":
        return submit_and_report("gesture", {"kind": args.kind}, wait)
    if args.cmd == "blink":
        return submit_and_report("blink", {}, wait)
    if args.cmd == "status":
        return submit_and_report("status", {}, wait)
    if args.cmd == "snapshot":
        return submit_and_report("snapshot", {"note": args.note}, wait)

    return 1


if __name__ == "__main__":
    sys.exit(main())
