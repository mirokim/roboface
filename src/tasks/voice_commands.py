"""음성 트리거 시스템 명령 — STT 결과에서 특정 키워드 들으면 동작.

ambient_listener.add_system_handler로 등록. consumed 발화는 conversation_log
/perception에 안 들어감 — agent가 "왜 그 말 했지?" 응답하는 거 차단.

현재 지원:
- "디버그 모드" → nmcli connection up jhS26u (폰 테더링 wifi 전환).
  실패 시 WIFI_FALLBACK_CHAIN 순차 시도.
- "셧다운" / "꺼" / "전원 꺼" → systemctl poweroff (안전 종료).
  confirm 패턴: 첫 발화로 10초 pending 진입(WORRIED), 둘째 발화로 실행.
  "취소" 또는 timeout 시 해제. STT false positive 한 번으론 종료 안 됨.

권한:
- NetworkManager: /etc/polkit-1/rules.d/50-nm-roboface.rules
- poweroff: /etc/polkit-1/rules.d/50-shutdown-roboface.rules
  (둘 다 repo 외부 설치 — robot 한 번만)
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

# 셧다운 confirm 윈도우 — 첫 발화 후 이 시간 안에 두 번째 발화 들으면 실행.
# 짧으면 사용자가 답할 시간 부족, 길면 우연한 두 번째 발화 위험.
_SHUTDOWN_CONFIRM_SEC = 10.0
# 셧다운 트리거 키워드 — _normalize 후 substring 매칭
_SHUTDOWN_TRIGGERS = ("셧다운", "shutdown", "전원꺼", "전원종료")
# 단독 "꺼" 같은 짧은 단어는 false positive 위험 너무 커서 제외 — "전원꺼"만
_SHUTDOWN_CANCEL_TRIGGERS = ("취소", "cancel", "아니야", "끄지마")


def _normalize(text: str) -> str:
    """공백/문장부호 제거 + 소문자. STT 출력 표기 다양성(.../!/공백) 흡수."""
    s = "".join(text.lower().split())
    return s.rstrip(".!?,~")


class VoiceCommandHandler:
    """ambient_listener.handlers에 등록되는 callable. 매 transcript 검사."""

    def __init__(self, face: FaceState) -> None:
        self.face = face
        self._last_triggered_at: dict[str, float] = {}
        # 셧다운 confirm 대기 시각 — 0이면 pending 아님
        self._shutdown_pending_until: float = 0.0

    async def __call__(self, text: str) -> bool:
        """ambient_listener system handler — True 반환 시 'consumed'.

        consumed 발화는 conversation_log/perception에 안 들어감 — agent가
        시스템 명령에 자연어로 반응하는 거 차단.
        """
        normalized = _normalize(text)
        now = time.time()

        # 1) 셧다운 confirm 대기 중이면 그 결정 먼저 — 다른 명령보다 우선.
        if self._shutdown_pending_until > 0 and now < self._shutdown_pending_until:
            if any(t in normalized for t in _SHUTDOWN_CANCEL_TRIGGERS):
                log.info(f'셧다운 취소 (text="{text}")')
                self._shutdown_pending_until = 0.0
                self.face.apply_expression(expr.CONTENT)
                self.face.show_speech("안 끄지", 2.0)
                return True
            if any(t in normalized for t in _SHUTDOWN_TRIGGERS):
                log.info(f'🛑 셧다운 확정 — text="{text}"')
                self._shutdown_pending_until = 0.0
                await self._perform_shutdown()
                return True
            # 그 외 발화는 confirm 윈도우 무시 — pending 그대로 (timeout 자연 해제)
            return False
        # pending timeout 자동 정리
        if self._shutdown_pending_until > 0 and now >= self._shutdown_pending_until:
            log.info("셧다운 confirm timeout — 자동 취소")
            self._shutdown_pending_until = 0.0
            self.face.show_speech("(셧다운 취소됨)", 2.0)

        # 2) 디버그 모드
        if "디버그모드" in normalized or "debugmode" in normalized:
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

        # 3) 셧다운 — 첫 발화면 confirm 대기 진입
        if any(t in normalized for t in _SHUTDOWN_TRIGGERS):
            log.info(f'🛑 셧다운 1단계 (confirm 대기) — text="{text}"')
            self._shutdown_pending_until = now + _SHUTDOWN_CONFIRM_SEC
            self.face.apply_expression(expr.WORRIED)
            self.face.show_speech(
                f"정말 끌까? 다시 '셧다운' (취소: '취소') · "
                f"{int(_SHUTDOWN_CONFIRM_SEC)}초",
                _SHUTDOWN_CONFIRM_SEC,
            )
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

    async def _perform_shutdown(self) -> None:
        """systemctl poweroff — polkit 권한으로 sudo 없이 실행.

        Pi5 안전 종료: systemd가 service 정리 → filesystem sync → power off.
        실행 직전 face/말풍선으로 사용자에게 알림 (전원 코드 제거 안내 포함).
        """
        log.info("🛑 셧다운 실행 — systemctl poweroff")
        self.face.apply_expression(expr.SLEEPY)
        self.face.show_speech("안녕… 곧 꺼져. 코드 뽑아도 돼.", 8.0)
        # systemd가 종료 시퀀스 시작 — service stop → sync → power off.
        # 1~3초 정도면 fs sync 끝나고 power 차단.
        await asyncio.sleep(1.0)
        try:
            proc = await asyncio.create_subprocess_exec(
                "systemctl", "poweroff",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=10.0,
            )
            if proc.returncode != 0:
                err = stderr_b.decode("utf-8", errors="ignore").strip()[:120]
                log.error(f"systemctl poweroff 실패 (rc={proc.returncode}): {err}")
                self.face.apply_expression(expr.WORRIED)
                self.face.show_speech(f"종료 실패: {err[:40]}", 5.0)
        except asyncio.TimeoutError:
            log.warning("systemctl poweroff timeout — 이미 종료 진행 중일 수 있음")
        except FileNotFoundError:
            log.error("systemctl not found")
            self.face.apply_expression(expr.WORRIED)
            self.face.show_speech("systemctl 없음", 3.0)
        except Exception as e:
            log.error(f"shutdown 에러: {e}")
            self.face.apply_expression(expr.WORRIED)
            self.face.show_speech(f"에러: {e}", 3.0)

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
