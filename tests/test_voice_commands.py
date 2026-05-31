"""voice_commands.VoiceCommandHandler — 트리거 매칭 + nmcli 호출."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from src.tasks.voice_commands import (
    PHONE_TETHER_SSID,
    VoiceCommandHandler,
    _normalize,
)


class _FaceStub:
    def __init__(self) -> None:
        self.expression_name = "neutral"
        self.speech_calls: list[tuple[str, float]] = []

    class _Expr:
        def __init__(self, name: str) -> None:
            self.name = name

    def __post_init__(self) -> None:
        pass

    @property
    def expression(self):
        return self._Expr(self.expression_name)

    def apply_expression(self, exp) -> None:
        self.expression_name = exp.name

    def show_speech(self, text: str, duration_sec: float) -> None:
        self.speech_calls.append((text, duration_sec))


def test_normalize_strips_whitespace_and_punct():
    assert _normalize("디버그 모드") == "디버그모드"
    assert _normalize("디버그 모드.") == "디버그모드"
    assert _normalize("디버그 모드!!!") == "디버그모드"
    assert _normalize(" 디버그모드 ") == "디버그모드"


def test_trigger_matches_various_phrasings():
    """'디버그 모드입니다' 등 추가 텍스트와 함께 와도 매칭."""
    face = _FaceStub()
    handler = VoiceCommandHandler(face)

    with patch.object(
        VoiceCommandHandler, "_run_nmcli_up",
        return_value=(0, "OK", ""),
    ) as m:
        asyncio.run(handler("디버그 모드"))
        asyncio.run(handler("이거 디버그모드 시작"))  # cooldown으로 두번째는 skip
    # 첫번째 호출만 nmcli 발동
    assert m.call_count == 1


def test_no_trigger_for_unrelated_text():
    face = _FaceStub()
    handler = VoiceCommandHandler(face)
    with patch.object(
        VoiceCommandHandler, "_run_nmcli_up",
        return_value=(0, "OK", ""),
    ) as m:
        asyncio.run(handler("디버깅이 어려워"))
        asyncio.run(handler("오늘 점심 뭐 먹지"))
    assert m.call_count == 0


def test_success_path_sets_happy_and_message():
    face = _FaceStub()
    handler = VoiceCommandHandler(face)
    with patch.object(
        VoiceCommandHandler, "_run_nmcli_up",
        return_value=(0, "OK", ""),
    ):
        asyncio.run(handler("디버그 모드"))
    assert face.expression_name == "happy"
    # FOCUSED 표시 → HAPPY 전환 도중 둘 다 show_speech 발동
    texts = [c[0] for c in face.speech_calls]
    assert any(PHONE_TETHER_SSID in t for t in texts)
    assert any("연결" in t for t in texts)


def test_fallback_chain_tries_second_wifi_on_phone_failure():
    """폰 실패 시 fallback chain의 첫 wifi로 자동 전환."""
    from src.tasks.voice_commands import WIFI_FALLBACK_CHAIN

    face = _FaceStub()
    handler = VoiceCommandHandler(face)

    call_log: list[str] = []

    async def fake_up(ssid, timeout=15.0):
        call_log.append(ssid)
        if ssid == PHONE_TETHER_SSID:
            return (1, "", "Error: Wi-Fi network could not be found")
        # 첫 fallback 성공
        if ssid == WIFI_FALLBACK_CHAIN[0]:
            return (0, "Connection activated", "")
        return (1, "", "Error: timeout")

    with patch.object(VoiceCommandHandler, "_run_nmcli_up", side_effect=fake_up):
        asyncio.run(handler("디버그 모드"))

    # 폰 먼저 시도, 그 다음 첫 fallback
    assert call_log[0] == PHONE_TETHER_SSID
    assert call_log[1] == WIFI_FALLBACK_CHAIN[0]
    # 첫 fallback 성공 → CONTENT
    assert face.expression_name == "content"
    texts = [c[0] for c in face.speech_calls]
    assert any(WIFI_FALLBACK_CHAIN[0] in t for t in texts)


def test_all_wifi_fail_gives_worried():
    """폰 + 모든 fallback 다 실패 시 WORRIED + 실패 메시지."""
    face = _FaceStub()
    handler = VoiceCommandHandler(face)

    with patch.object(
        VoiceCommandHandler, "_run_nmcli_up",
        return_value=(1, "", "Error: not found"),
    ):
        asyncio.run(handler("디버그 모드"))

    assert face.expression_name == "worried"
    texts = [c[0] for c in face.speech_calls]
    assert any("전부 실패" in t or "wifi 연결" in t for t in texts)


def test_cooldown_blocks_second_trigger_within_30s():
    face = _FaceStub()
    handler = VoiceCommandHandler(face)
    with patch.object(
        VoiceCommandHandler, "_run_nmcli_up",
        return_value=(0, "OK", ""),
    ) as m:
        asyncio.run(handler("디버그 모드"))
        # 즉시 다시 — cooldown
        asyncio.run(handler("디버그 모드"))
    assert m.call_count == 1
