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

    async def __call__(self, text: str) -> None:
        """ambient_listener handler 시그니처 (async def handler(text))."""
        normalized = _normalize(text)
        # "디버그 모드", "디버그모드", "디버그 모드입니다" 등 모두 매칭
        if "디버그모드" in normalized or "debugmode" in normalized:
            now = time.time()
            last = self._last_triggered_at.get("debug_mode", 0.0)
            if now - last < _COOLDOWN_SEC:
                log.info(f"디버그 모드 트리거 cooldown ({now - last:.0f}s < {_COOLDOWN_SEC})")
                return
            self._last_triggered_at["debug_mode"] = now
            log.info(f'🛠 디버그 모드 트리거 — text="{text}"')
            await self._connect_phone_tether()

    async def _connect_phone_tether(self) -> None:
        """nmcli connection up — 비동기 subprocess, timeout 15s."""
        self.face.apply_expression(expr.FOCUSED)
        self.face.show_speech(f"디버그 모드: {PHONE_TETHER_SSID} 연결 중…", 5.0)

        rc, stdout, stderr = await self._run_nmcli_up(PHONE_TETHER_SSID, timeout=15.0)
        if rc == 0:
            log.info(f"📶 {PHONE_TETHER_SSID} 연결 성공")
            self.face.apply_expression(expr.HAPPY)
            self.face.show_speech("폰 테더링 연결!", 3.0)
        else:
            err = (stderr or stdout or "").strip()[:100]
            # 가장 흔한 케이스: 폰 핫스팟 꺼져있음 → "Wi-Fi network could not be found"
            if "could not be found" in err.lower() or "not found" in err.lower():
                msg = f"{PHONE_TETHER_SSID} 안 보임 (폰 켜야)"
            else:
                msg = f"연결 실패 (rc={rc})"
            log.warning(f"nmcli 실패: rc={rc}, err={err}")
            self.face.apply_expression(expr.WORRIED)
            self.face.show_speech(msg, 3.0)

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
