"""AmbientListener system_handler — consumed 발화가 일반 흐름 차단하는지."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from src.tasks.ambient_listener import AmbientListener


class _MockSTT:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts

    async def stream(self):
        for t in self.texts:
            yield t


class _Perception:
    def __init__(self) -> None:
        self.last_user_speech_at: float = 0.0


def test_system_consumed_skips_log_and_handlers(monkeypatch):
    """system handler가 True 반환하면 memory.log_user / perception / 일반 handler 모두 skip."""
    perception = _Perception()
    stt = _MockSTT(["디버그 모드", "오늘 점심 뭐"])
    al = AmbientListener(stt=stt, perception=perception)

    general_calls: list[str] = []

    async def general_handler(text: str) -> None:
        general_calls.append(text)

    sys_consumed_texts: list[str] = []

    async def sys_handler(text: str) -> bool:
        if "디버그" in text:
            sys_consumed_texts.append(text)
            return True
        return False

    al.add_handler(general_handler)
    al.add_system_handler(sys_handler)

    log_calls: list[tuple[str, str]] = []

    def fake_log_user(text, kind="ambient"):
        log_calls.append((text, kind))

    # memory.log_user 모킹 — 진짜 DB 접근 막음
    monkeypatch.setattr("src.tasks.ambient_listener.memory.log_user", fake_log_user)

    asyncio.run(al.run())

    # "디버그 모드"는 sys handler가 consume
    assert sys_consumed_texts == ["디버그 모드"]
    # 일반 handler는 "오늘 점심 뭐"만 받음 (디버그 모드는 skip)
    assert general_calls == ["오늘 점심 뭐"]
    # memory.log_user도 일반 발화만 (디버그 모드 X)
    assert log_calls == [("오늘 점심 뭐", "ambient")]
    # perception도 일반 발화 시점만 갱신
    assert perception.last_user_speech_at > 0


def test_system_handler_falsy_passes_through(monkeypatch):
    """system handler가 False/None 반환하면 일반 흐름 그대로 진행."""
    perception = _Perception()
    stt = _MockSTT(["안녕"])
    al = AmbientListener(stt=stt, perception=perception)

    async def sys_handler(text: str) -> bool:
        return False   # 항상 pass through

    al.add_system_handler(sys_handler)

    general_calls: list[str] = []

    async def general_handler(text: str) -> None:
        general_calls.append(text)

    al.add_handler(general_handler)
    log_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "src.tasks.ambient_listener.memory.log_user",
        lambda text, kind="ambient": log_calls.append((text, kind)),
    )

    asyncio.run(al.run())

    assert general_calls == ["안녕"]
    assert log_calls == [("안녕", "ambient")]


def test_system_handler_exception_does_not_consume(monkeypatch):
    """system handler가 raise해도 일반 흐름 계속 — 시스템 명령 깨져도 사용자 발화 보존."""
    perception = _Perception()
    stt = _MockSTT(["안녕"])
    al = AmbientListener(stt=stt, perception=perception)

    async def bad_sys_handler(text: str) -> bool:
        raise RuntimeError("nmcli 깨짐")

    al.add_system_handler(bad_sys_handler)

    general_calls: list[str] = []

    async def general_handler(text: str) -> None:
        general_calls.append(text)

    al.add_handler(general_handler)
    log_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "src.tasks.ambient_listener.memory.log_user",
        lambda text, kind="ambient": log_calls.append((text, kind)),
    )

    asyncio.run(al.run())

    # 예외가 consume으로 이어지면 안 됨 — 일반 흐름 진행
    assert general_calls == ["안녕"]
    assert log_calls == [("안녕", "ambient")]
