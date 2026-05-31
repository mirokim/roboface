"""음성 트리거 시스템 명령 — STT 결과에서 특정 키워드 들으면 동작.

ambient_listener.add_system_handler로 등록. consumed 발화는 conversation_log
/perception에 안 들어감 — agent가 "왜 그 말 했지?" 응답하는 거 차단.

현재 지원:
- "디버그 모드" → nmcli connection up jhS26u (폰 테더링 wifi 전환).
  실패 시 WIFI_FALLBACK_CHAIN 순차 시도.
- "셧다운" / "전원 꺼" → systemctl poweroff (안전 종료).
  confirm 패턴: 첫 발화로 10초 pending 진입(WORRIED), 둘째 발화로 실행.
- "재시작" / "다시 시작" / "restart" → systemctl restart roboface (service만).
  코드 변경 반영 / hang 회복용. 셧다운과 동일 10초 confirm 패턴.
- "날씨 알려줘" / "날씨 어때" / "오늘 날씨" → WeatherClient.snapshot()
  결과(예: "분당 맑음 18°C · 습도 45%") LCD에 8초 표시. 30분 캐시 hit면 즉시.

권한:
- NetworkManager: /etc/polkit-1/rules.d/50-nm-roboface.rules
- poweroff: /etc/polkit-1/rules.d/50-shutdown-roboface.rules
- systemctl restart roboface: /etc/polkit-1/rules.d/50-systemd-roboface.rules
  (모두 repo 외부 설치 — robot 한 번만)
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

# 재시작 — service만 다시 (systemctl restart roboface). 코드 변경 반영용.
# 셧다운과 동일 10초 confirm 패턴.
_RESTART_CONFIRM_SEC = 10.0
_RESTART_TRIGGERS = ("재시작", "다시시작", "리스타트", "restart", "리셋")
# cancel은 셧다운과 공유 (취소/cancel/아니야) — "끄지마" 대신 일반 표현만 매칭
_RESTART_CANCEL_TRIGGERS = ("취소", "cancel", "아니야", "하지마")

# 날씨 — 명시적 패턴만 (단독 "날씨"는 일반 대화 "날씨 좋네"도 잡아 noisy).
# _normalize 후 공백/문장부호 제거 상태 기준.
_WEATHER_TRIGGERS = (
    "날씨알려", "날씨어때", "날씨가어때", "날씨좀",
    "오늘날씨", "지금날씨", "weather",
)
# 내일 forecast — "내일" 단어 자체가 날씨 문맥 외엔 잘 안 쓰여 단순 substring OK
_TOMORROW_WEATHER_TRIGGERS = (
    "내일날씨", "내일은날씨", "내일기온", "내일은어때",
)
# 같은 질문 빠른 반복 막음 — 짧게 (캐시라 cost 없지만 LCD 깜빡 방지)
_WEATHER_COOLDOWN_SEC = 5.0


def _normalize(text: str) -> str:
    """공백/문장부호 제거 + 소문자. STT 출력 표기 다양성(.../!/공백) 흡수."""
    s = "".join(text.lower().split())
    return s.rstrip(".!?,~")


class VoiceCommandHandler:
    """ambient_listener.handlers에 등록되는 callable. 매 transcript 검사."""

    def __init__(self, face: FaceState) -> None:
        self.face = face
        self._last_triggered_at: dict[str, float] = {}
        # confirm 대기 시각 — 0이면 pending 아님. 명령별 독립.
        self._shutdown_pending_until: float = 0.0
        self._restart_pending_until: float = 0.0

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

        # 1b) 재시작 confirm 대기 중 — 셧다운과 동일 패턴
        if self._restart_pending_until > 0 and now < self._restart_pending_until:
            if any(t in normalized for t in _RESTART_CANCEL_TRIGGERS):
                log.info(f'재시작 취소 (text="{text}")')
                self._restart_pending_until = 0.0
                self.face.apply_expression(expr.CONTENT)
                self.face.show_speech("그냥 둘게", 2.0)
                return True
            if any(t in normalized for t in _RESTART_TRIGGERS):
                log.info(f'🔄 재시작 확정 — text="{text}"')
                self._restart_pending_until = 0.0
                await self._perform_restart()
                return True
            return False
        if self._restart_pending_until > 0 and now >= self._restart_pending_until:
            log.info("재시작 confirm timeout — 자동 취소")
            self._restart_pending_until = 0.0
            self.face.show_speech("(재시작 취소됨)", 2.0)

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

        # 3b) 재시작 — 셧다운보다 덜 위험하지만 confirm 동일 (false positive 보호)
        if any(t in normalized for t in _RESTART_TRIGGERS):
            log.info(f'🔄 재시작 1단계 (confirm 대기) — text="{text}"')
            self._restart_pending_until = now + _RESTART_CONFIRM_SEC
            self.face.apply_expression(expr.FOCUSED)
            self.face.show_speech(
                f"재시작? 다시 '재시작' (취소: '취소') · "
                f"{int(_RESTART_CONFIRM_SEC)}초",
                _RESTART_CONFIRM_SEC,
            )
            return True

        # 4) 내일 날씨 — forecast endpoint (현재 weather보다 먼저 체크: specific 우선)
        if any(t in normalized for t in _TOMORROW_WEATHER_TRIGGERS):
            last = self._last_triggered_at.get("weather_tomorrow", 0.0)
            if now - last < _WEATHER_COOLDOWN_SEC:
                log.info(
                    f"내일 날씨 cooldown ({now - last:.0f}s < {_WEATHER_COOLDOWN_SEC})"
                )
                return True
            self._last_triggered_at["weather_tomorrow"] = now
            log.info(f'🌤 내일 날씨 요청 — text="{text}"')
            await self._announce_tomorrow_weather()
            return True

        # 5) 오늘/현재 날씨 — WeatherClient.snapshot() LCD 표시
        if any(t in normalized for t in _WEATHER_TRIGGERS):
            last = self._last_triggered_at.get("weather", 0.0)
            if now - last < _WEATHER_COOLDOWN_SEC:
                log.info(
                    f"날씨 cooldown ({now - last:.0f}s < {_WEATHER_COOLDOWN_SEC})"
                )
                return True
            self._last_triggered_at["weather"] = now
            log.info(f'🌤 날씨 요청 — text="{text}"')
            await self._announce_weather()
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

    async def _announce_weather(self) -> None:
        """현재 날씨 LCD에 8초 표시. WeatherClient 30분 캐시 활용."""
        try:
            from src.integrations.weather import get_client
            snap = await get_client().snapshot()
        except Exception as e:
            log.warning(f"weather snapshot 실패: {e}")
            self.face.apply_expression(expr.WORRIED)
            self.face.show_speech("날씨 정보 가져오기 실패", 3.0)
            return
        if snap is None:
            log.info("weather snapshot None — OPENWEATHER_API_KEY 미설정")
            self.face.apply_expression(expr.WORRIED)
            self.face.show_speech("날씨 정보 없음 (API 키 X)", 3.0)
            return
        line = snap.one_liner()
        log.info(f"🌤 {line}")
        self.face.apply_expression(expr.CONTENT)
        self.face.show_speech(line, 8.0)

    async def _announce_tomorrow_weather(self) -> None:
        """내일 날씨 LCD에 8초 표시. 1시간 캐시 + 5일 forecast endpoint."""
        try:
            from src.integrations.weather import get_client
            snap = await get_client().forecast_for_tomorrow()
        except Exception as e:
            log.warning(f"tomorrow forecast 실패: {e}")
            self.face.apply_expression(expr.WORRIED)
            self.face.show_speech("내일 날씨 가져오기 실패", 3.0)
            return
        if snap is None:
            log.info("tomorrow forecast None — 키 없거나 fetch 실패")
            self.face.apply_expression(expr.WORRIED)
            self.face.show_speech("내일 날씨 정보 없음", 3.0)
            return
        line = snap.one_liner()
        log.info(f"🌤 {line}")
        self.face.apply_expression(expr.CONTENT)
        self.face.show_speech(line, 8.0)

    async def _perform_restart(self) -> None:
        """systemctl restart roboface — service만 다시. polkit으로 sudo 없이.

        실행하면 systemd가 SIGTERM 보내 main_robot graceful shutdown 후
        새 process 즉시 spawn. 사용자 시점엔 5초 정도 LCD 꺼졌다가 복귀.

        이 메서드 자체는 systemctl 호출 후 곧 자기 process가 kill됨 —
        post-restart 로깅은 불가능. 호출 직전에만 LCD 표시.
        """
        log.info("🔄 재시작 실행 — systemctl restart roboface")
        self.face.apply_expression(expr.SLEEPY)
        self.face.show_speech("다시 시작할게… 잠깐만.", 6.0)
        await asyncio.sleep(0.5)   # LCD 잠시 보여줌
        try:
            proc = await asyncio.create_subprocess_exec(
                "systemctl", "restart", "roboface",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            # restart는 곧 자기 process kill — wait_for는 일반적으로 SIGTERM 받기 전에 끝남.
            # 짧은 timeout으로 만약 명령 자체 실패하면 알 수 있게.
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=5.0,
                )
                if proc.returncode != 0:
                    err = stderr_b.decode("utf-8", errors="ignore").strip()[:120]
                    log.error(f"systemctl restart 실패 (rc={proc.returncode}): {err}")
                    self.face.apply_expression(expr.WORRIED)
                    self.face.show_speech(f"재시작 실패: {err[:40]}", 5.0)
            except asyncio.TimeoutError:
                # 정상 케이스 — systemctl이 자기를 kill하는 중. 곧 SIGTERM 받음.
                log.info("systemctl restart 진행 중 — 곧 SIGTERM")
        except FileNotFoundError:
            log.error("systemctl not found")
            self.face.apply_expression(expr.WORRIED)
            self.face.show_speech("systemctl 없음", 3.0)
        except Exception as e:
            log.error(f"restart 에러: {e}")
            self.face.apply_expression(expr.WORRIED)
            self.face.show_speech(f"에러: {e}", 3.0)

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
