"""한국어 호칭/조사 헬퍼 — 이름 뒤 호격 조사 SSOT."""

from __future__ import annotations

import random


def vocative(name: str) -> str:
    """'미로'→'미로야', '철민'→'철민아', '주인님'→'주인님' (님/비한글은 조사 X)."""
    if not name:
        return ""
    last = name[-1]
    if name.endswith("님") or not ("가" <= last <= "힣"):
        return name
    return name + ("아" if (ord(last) - 0xAC00) % 28 else "야")


def name_prefix(name: str | None) -> str:
    """멘트 앞에 붙일 호칭 — '미로야, ' / '미로, ' / '미로! ' 중 무작위."""
    if not name:
        return ""
    return random.choice([f"{vocative(name)}, ", f"{name}, ", f"{name}! "])
