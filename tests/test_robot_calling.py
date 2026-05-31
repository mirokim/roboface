"""로봇 호명 감지 — "네모", "네모야" 등 호격 패턴."""

from __future__ import annotations

import asyncio
import time

import pytest

from src.tasks.ambient_listener import _is_calling_robot


# ─── 호명 매칭 ───

@pytest.mark.parametrize("text", [
    "네모",
    "네모야",
    "네모아",
    "네모야 안녕",
    "야 네모",
    "네모 뭐해",
    "안녕 네모",
    "네모 들어봐",
    "네모, 뭐 해?",
    "네모.",
])
def test_calling_matches(text):
    assert _is_calling_robot(text), f"호명 매칭 실패: {text!r}"


@pytest.mark.parametrize("text", [
    "네모난 박스",        # "네모"가 단어 prefix — 단어 단위 매칭이라 False
    "그 네모 영화 봤어",  # 중간 위치
    "이거 네모 모양이야",  # 중간 위치
    "안녕 로봇",          # "네모" 자체 없음
    "오늘 점심 뭐 먹지",
    "",
])
def test_not_calling(text):
    assert not _is_calling_robot(text), f"호명으로 잘못 매칭: {text!r}"


def test_calling_matches_with_punctuation_and_spacing():
    """공백/문장부호 변형 흡수 — 실제 STT 출력 형태."""
    for text in [" 네모야 ", "네모야!", "네모,", "네모?", "네모. 안녕"]:
        assert _is_calling_robot(text), f"매칭 실패: {text!r}"


# ─── ambient_listener integration ───

class _MockSTT:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts

    async def stream(self):
        for t in self.texts:
            yield t


class _Perception:
    last_user_speech_at: float = 0.0
    last_user_called_at: float = 0.0


def test_ambient_listener_marks_called_at(monkeypatch):
    """ambient_listener가 호명 발화 받으면 perception.last_user_called_at 갱신."""
    from src.tasks.ambient_listener import AmbientListener

    perception = _Perception()
    stt = _MockSTT(["네모야", "오늘 점심 뭐"])
    al = AmbientListener(stt=stt, perception=perception)
    monkeypatch.setattr(
        "src.tasks.ambient_listener.memory.log_user",
        lambda text, kind="ambient": None,
    )

    asyncio.run(al.run())

    # 호명 발화 후 called_at 갱신, 일반 발화는 안 갱신 (마지막은 일반이라 called_at은 처음 값 유지)
    assert perception.last_user_called_at > 0
    assert perception.last_user_speech_at >= perception.last_user_called_at


def test_ambient_listener_no_call_no_marking(monkeypatch):
    """호명 없는 일반 발화엔 called_at 안 갱신."""
    from src.tasks.ambient_listener import AmbientListener

    perception = _Perception()
    stt = _MockSTT(["오늘 점심 뭐 먹지"])
    al = AmbientListener(stt=stt, perception=perception)
    monkeypatch.setattr(
        "src.tasks.ambient_listener.memory.log_user",
        lambda text, kind="ambient": None,
    )

    asyncio.run(al.run())

    assert perception.last_user_called_at == 0.0
    assert perception.last_user_speech_at > 0
