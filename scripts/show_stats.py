"""현재 로봇 스탯 보기 — 디버그용.

사용: python -m scripts.show_stats
"""

from __future__ import annotations

from src.brain import memory, stats as robot_stats


def main() -> None:
    memory.init_db()
    s = robot_stats.get()
    print(f"energy   : {s.energy:5.1f} / 100")
    print(f"mood     : {s.mood:5.1f} / 100")
    print(f"social   : {s.social:5.1f} / 100")
    print(f"curiosity: {s.curiosity:5.1f} / 100")
    print(f"label    : {robot_stats.mood_label()}")
    suggested = robot_stats.suggested_expression()
    print(f"suggested expr: {suggested or '(없음)'}")


if __name__ == "__main__":
    main()
