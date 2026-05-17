"""roboface 외부 제어 CLI.

실행 중인 roboface 프로세스(main.py / main_robot.py)에 SQLite 큐로 명령 전송.
명령은 메인 프로세스의 command_executor가 1초 안에 처리.

사용:
    python scripts/robot_cli.py speak "안녕 미로야"
    python scripts/robot_cli.py speak "안녕!" --expression happy
    python scripts/robot_cli.py expression curious
    python scripts/robot_cli.py dance --beats 6 --bpm 130
    python scripts/robot_cli.py pose nod
    python scripts/robot_cli.py blink
    python scripts/robot_cli.py transition WATCHING
    python scripts/robot_cli.py status
    python scripts/robot_cli.py expressions    # 사용 가능한 표정 목록

--wait 옵션을 주면 명령 처리 완료까지 대기 후 result 출력.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 프로젝트 루트를 path에 추가 (스크립트 직접 실행 대응)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.brain import memory  # noqa: E402
from src.brain.state_machine import State  # noqa: E402
from src.face.expressions import EXPRESSIONS_BY_NAME  # noqa: E402


def _wait_for(cmd_id: int, timeout: float = 10.0) -> dict | None:
    """명령 처리 완료까지 폴링."""
    end = time.time() + timeout
    while time.time() < end:
        st = memory.get_command_status(cmd_id)
        if st and st["status"] in ("done", "failed"):
            return st
        time.sleep(0.2)
    return None


def _enqueue_and_report(cmd: str, args: dict, wait: bool) -> int:
    memory.init_db()
    cmd_id = memory.enqueue_command(cmd, args)
    print(f"# cmd #{cmd_id}: {cmd} {args}")
    if not wait:
        return 0
    st = _wait_for(cmd_id)
    if st is None:
        print("(timeout - 메인 프로세스가 안 떠있나?)")
        return 2
    print(f"status: {st['status']}")
    if st.get("result"):
        print(f"result: {st['result']}")
    return 0 if st["status"] == "done" else 1


def main() -> int:
    p = argparse.ArgumentParser(prog="robot_cli")
    p.add_argument("--wait", action="store_true",
                   help="처리 완료까지 대기 후 결과 출력")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("speak", help="발화")
    sp.add_argument("text")
    sp.add_argument("--expression", "-e", default=None)

    se = sub.add_parser("expression", help="표정만 변경")
    se.add_argument("name")

    sd = sub.add_parser("dance", help="짧은 댄스 (서보 필요)")
    sd.add_argument("--beats", type=int, default=4)
    sd.add_argument("--bpm", type=int, default=120)

    sp2 = sub.add_parser("pose", help="포즈 (서보 필요)")
    sp2.add_argument("kind",
                     choices=["nod", "shake", "greeting",
                              "tilt_curious", "look_around"])

    st = sub.add_parser("transition", help="상태 머신 강제 전이")
    st.add_argument("state", choices=[s.name for s in State])

    sg = sub.add_parser(
        "gesture",
        help="vision 우회해 sensor event 강제 emit (제스처 인식 downstream 검증)",
    )
    sg.add_argument(
        "kind",
        choices=["wave", "hands_up", "nod", "shake", "gaze",
                 "presence_new", "presence_left"],
    )

    sub.add_parser("blink", help="즉시 깜빡임")
    sub.add_parser("status", help="현재 상태 조회 (--wait 자동)")
    sub.add_parser("expressions", help="사용 가능한 표정 이름 목록")

    ss = sub.add_parser(
        "snapshot",
        help="카메라 프레임에 bbox+keypoints 그려서 /tmp/roboface_debug.jpg 저장",
    )
    ss.add_argument("--note", default="", help="이미지에 표시할 메모")

    args = p.parse_args()

    if args.cmd == "expressions":
        for name in sorted(EXPRESSIONS_BY_NAME):
            print(name)
        return 0

    wait = args.wait or args.cmd == "status"

    if args.cmd == "speak":
        return _enqueue_and_report(
            "speak", {"text": args.text, "expression": args.expression}, wait,
        )
    if args.cmd == "expression":
        return _enqueue_and_report("expression", {"name": args.name}, wait)
    if args.cmd == "dance":
        return _enqueue_and_report(
            "dance", {"beats": args.beats, "bpm": args.bpm}, wait,
        )
    if args.cmd == "pose":
        return _enqueue_and_report("pose", {"kind": args.kind}, wait)
    if args.cmd == "transition":
        return _enqueue_and_report("transition", {"state": args.state}, wait)
    if args.cmd == "gesture":
        return _enqueue_and_report("gesture", {"kind": args.kind}, wait)
    if args.cmd == "blink":
        return _enqueue_and_report("blink", {}, wait)
    if args.cmd == "snapshot":
        return _enqueue_and_report("snapshot", {"note": args.note}, wait)
    if args.cmd == "status":
        return _enqueue_and_report("status", {}, wait)

    return 1


if __name__ == "__main__":
    sys.exit(main())
