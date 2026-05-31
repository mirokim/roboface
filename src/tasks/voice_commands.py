"""음성 트리거 시스템 명령 — STT 결과에서 특정 키워드 들으면 동작.

ambient_listener의 add_handler로 등록. perception/agent 무관 즉시 실행.

현재 지원:
- "디버그 모드" → nmcli connection up jhS26u (폰 테더링 wifi 전환).
  회사 wifi(autoconnect off) 환경에서 폰 켜고 빠르게 robot을 폰 네트워크로
  넘겨 SSH/디버그 가능.

권한: NetworkManager 제어는 polkit으로 miro 사용자에 부여
(/etc/polkit-1/rules.d/50-nm-roboface.rules — repo 외부 설치).
"""

from __future__ import annotations

import asyncio
import time

from src.face import expressions as expr
from src.face.renderer import FaceState
from src.utils.logger import get_logger

log = get_logger("voice_cmd")


# 폰 테더링 wifi 프로파일 이름 (NetworkManager connection name)
PHONE_TETHER_SSID = "jhS26u"

# 폴백 chain — 폰 핫스팟 없을 때 순차 시도할 wifi 프로파일.
# nmcli connection show로 본 robot에 등록된 이름 그대로. 새 wifi 추가하면
# 여기 append. priority 순서: 폰 > 안정적인 집/사무실 wifi > 공용 wifi.
WIFI_FALLBACK_CHAIN = [
    "netplan-wlan0-bbang5G",   # 평소 메인 wifi
    "SG_Open",                  # 공용 백업
]

# 같은 명령 빠르게 여러 번 발동 방지
_COOLDOWN_SEC = 30.0


def _normalize(text: str) -> str:
    """공백/문장부호 제거 + 소문자. STT 출력 표기 다양성(.../!/공백) 흡수."""
    s = "".join(text.lower().split())
    return s.rstrip(".!?,~")


class VoiceCommandHandler:
    """ambient_listener.handlers에 등록되는 callable. 매 transcript 검사."""

    def __init__(self, face: FaceState) -> None:
        self.face = face
        self._last_triggered_at: dict[str, float] = {}

    async def __call__(self, text: str) -> bool:
        """ambient_listener system handler — True 반환 시 'consumed'.

        consumed 발화는 conversation_log/perception에 안 들어감 — agent가
        시스템 명령에 자연어로 반응하는 거 차단.
        """
        normalized = _normalize(text)
        # "디버그 모드", "디버그모드", "디버그 모드입니다" 등 모두 매칭
        if "디버그모드" in normalized or "debugmode" in normalized:
            now = time.time()
            last = self._last_triggered_at.get("debug_mode", 0.0)
            if now - last < _COOLDOWN_SEC:
                log.info(
                    f"디버그 모드 트리거 cooldown "
                    f"({now - last:.0f}s < {_COOLDOWN_SEC})"
                )
                return True   # cooldown이라 동작 안 했어도 명령 자체는 consumed
            self._last_triggered_at["debug_mode"] = now
            log.info(f'🛠 디버그 모드 트리거 — text="{text}"')
            await self._connect_phone_tether()
            return True
        return False

    async def _connect_phone_tether(self) -> None:
        """폰 테더링 우선 시도 → 실패 시 fallback chain 순차 시도.

        성공한 첫 wifi에서 종료. 모두 실패면 WORRIED + 실패 메시지.
        """
        self.face.apply_expression(expr.FOCUSED)
        self.face.show_speech(f"디버그 모드: {PHONE_TETHER_SSID} 연결 중…", 5.0)

        # 1차 — 폰 테더링
        rc, _, stderr = await self._run_nmcli_up(PHONE_TETHER_SSID, timeout=12.0)
        if rc == 0:
            log.info(f"📶 {PHONE_TETHER_SSID} 연결 성공 (폰 테더링)")
            self.face.apply_expression(expr.HAPPY)
            self.face.show_speech("폰 테더링 연결!", 3.0)
            return

        # 2차 이후 — fallback chain
        phone_err = (stderr or "").strip()[:80]
        log.info(f"폰 테더링 실패 (rc={rc}): {phone_err} — 폴백 시도")
        self.face.show_speech("폰 안 보임 — 다른 wifi 시도…", 4.0)

        for ssid in WIFI_FALLBACK_CHAIN:
            rc, _, stderr = await self._run_nmcli_up(ssid, timeout=12.0)
            if rc == 0:
                log.info(f"📶 폴백 wifi 연결: {ssid}")
                self.face.apply_expression(expr.CONTENT)
                self.face.show_speech(f"{ssid} 연결됨", 3.0)
                return
            err = (stderr or "").strip()[:80]
            log.info(f"폴백 {ssid} 실패 (rc={rc}): {err}")

        # 전부 실패
        log.warning("모든 wifi 연결 실패")
        self.face.apply_expression(expr.WORRIED)
        self.face.show_speech("wifi 연결 전부 실패", 3.0)

    @staticmethod
    async def _run_nmcli_up(
        ssid: str, timeout: float = 15.0,
    ) -> tuple[int, str, str]:
        """nmcli connection up SSID — (returncode, stdout, stderr)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "nmcli", "connection", "up", ssid,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return 127, "", "nmcli not installed"
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return 124, "", f"timeout after {timeout}s"
        return (
            proc.returncode if proc.returncode is not None else -1,
            stdout_b.decode("utf-8", errors="ignore"),
            stderr_b.decode("utf-8", errors="ignore"),
        )
