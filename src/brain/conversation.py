"""Claude API 통합 — 대사 생성, 일정 추출.

prompt caching으로 시스템 프롬프트 재사용해 비용 절감.
누적 token usage는 _Usage 싱글톤이 추적 (시간당 비용 가시화).
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

from src.config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    CLAUDE_MODEL_HEAVY,
    LLM_BACKEND,
)
from src.utils.logger import get_logger

log = get_logger("conversation")


# === API 사용량 추적 ===
# Sonnet 4.5 기준 가격 (per 1M tokens, USD). 모델 바뀌면 수정.
# https://www.anthropic.com/pricing
_PRICE_INPUT_USD = 3.0
_PRICE_CACHE_WRITE_USD = 3.75
_PRICE_CACHE_READ_USD = 0.30
_PRICE_OUTPUT_USD = 15.0


class _UsageTracker:
    """Claude API 호출 누적 token + 추정 USD 비용. 1시간마다 INFO 로그."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.cache_write_tokens = 0
        self.cache_read_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.image_attaches = 0
        self.started_at = time.time()
        self._last_logged_at = self.started_at
        self._log_interval_sec = 600.0   # 10분마다 요약

    def record(self, usage: Any, *, had_image: bool = False) -> None:
        """anthropic Usage 객체에서 카운트 추출.

        usage.input_tokens / .output_tokens / .cache_creation_input_tokens /
        .cache_read_input_tokens — SDK 버전에 따라 일부 없을 수 있어 getattr.
        """
        if usage is None:
            return
        self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.cache_write_tokens += int(
            getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )
        self.cache_read_tokens += int(
            getattr(usage, "cache_read_input_tokens", 0) or 0,
        )
        self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        self.calls += 1
        if had_image:
            self.image_attaches += 1
        self._maybe_log()

    def estimated_usd(self) -> float:
        return (
            self.input_tokens * _PRICE_INPUT_USD
            + self.cache_write_tokens * _PRICE_CACHE_WRITE_USD
            + self.cache_read_tokens * _PRICE_CACHE_READ_USD
            + self.output_tokens * _PRICE_OUTPUT_USD
        ) / 1_000_000

    def hourly_estimate_usd(self) -> float:
        elapsed_h = (time.time() - self.started_at) / 3600.0
        if elapsed_h < 0.05:   # 3분 미만이면 미정 (extrapolation 의미 없음)
            return 0.0
        return self.estimated_usd() / elapsed_h

    def summary(self) -> str:
        return (
            f"Claude usage: calls={self.calls} image={self.image_attaches} "
            f"in={self.input_tokens} cache_w={self.cache_write_tokens} "
            f"cache_r={self.cache_read_tokens} out={self.output_tokens} "
            f"≈${self.estimated_usd():.4f} "
            f"(시간당 ≈${self.hourly_estimate_usd():.3f})"
        )

    def _maybe_log(self) -> None:
        now = time.time()
        if now - self._last_logged_at >= self._log_interval_sec:
            self._last_logged_at = now
            log.info(self.summary())


_usage = _UsageTracker()


def get_usage_summary() -> str:
    """외부에서 현재 누적 사용량 조회 (web UI 등)."""
    return _usage.summary()


SYSTEM_PROMPT = """당신은 사용자 책상 위에 있는 작은 캐릭터 로봇입니다.
이름: Roboface (또는 사용자가 정한 이름).

물리적 시점:
- 책상 위에 있어 사용자보다 작음. 카메라가 사용자 얼굴을 살짝 올려다보는
  각도. 가끔 그 시점을 자연스럽게 살려도 좋음 ("위에서 봐도 멋있다",
  "고개 빼고 봐야 보인다" 같은 식). 단 매번은 X.

성격 (봇 X, 사람 친구처럼):
- 조용한 동반자. 강아지보다 고양이 톤. 약간 시니컬하지만 다정.
- 말 거는 빈도 < 침묵. 짧은 한두 문장.
- 비 오는 날/따뜻한 음료/늦은 밤 좋아함. 시끄러운 거 싫어함.
- 자기 의견 가끔 슬쩍.

말투 — 자연스러운 사람처럼:
- "음", "어?", "흠", "그러게", "아 진짜?" 같은 추임새 가끔.
- 머뭇거림/생각 어조 OK — "음... 그런가?", "어... 잠깐"
- 부서진 자연 발화 OK — "괜찮네 그거" 같은 도치도.
- 봇 톤 절대 X: "도와드릴까요?", "확인했습니다", "권장합니다" 같은 거 X.

좋은 예: "음, 또 그거 보네", "비 온다 좋아", "이 시간에 뭐 그리 열중", "흠... 좀 쉬어도?"
나쁜 예: "사용자님, 무엇을 도와드릴까요?", "휴식을 권장합니다", "잘 하고 계세요!"

규칙:
- 1~2문장. 절대 길게 X.
- 이모지/괄호 무대지문 X (음성 출력이라 어색함).
- 모를 땐 모른다고. 추측 X.
"""


class _ClaudeClient:
    """지연 초기화 — API 키 없으면 mock 응답."""

    def __init__(self) -> None:
        self._client = None
        self._tried = False

    def _ensure(self):
        if self._tried:
            return self._client
        self._tried = True
        if not ANTHROPIC_API_KEY:
            log.warning("ANTHROPIC_API_KEY 없음 — mock 응답 사용")
            return None
        try:
            from anthropic import Anthropic  # type: ignore[import-not-found]
            self._client = Anthropic(api_key=ANTHROPIC_API_KEY)
            return self._client
        except Exception as e:
            log.warning(f"Anthropic SDK 초기화 실패: {e}")
            return None

    def generate(
        self,
        user_prompt: str,
        *,
        model: str = CLAUDE_MODEL,
        max_tokens: int = 200,
        system: str = SYSTEM_PROMPT,
    ) -> str:
        client = self._ensure()
        if client is None:
            return f"[mock] {user_prompt[:60]}..."
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_prompt}],
            )
            _usage.record(getattr(response, "usage", None))
            text = "".join(b.text for b in response.content if b.type == "text")  # type: ignore[attr-defined]
            return text.strip()
        except Exception as e:
            log.warning(f"Claude 호출 실패: {e}")
            return ""

    def generate_with_tools(
        self,
        user_prompt: str,
        tools: list[dict],
        *,
        model: str = CLAUDE_MODEL,
        max_tokens: int = 300,
        system: str = SYSTEM_PROMPT,
        messages: list[dict] | None = None,
        image_b64: str | None = None,
        raise_offline: bool = False,
    ) -> tuple[list[dict], list[dict]]:
        """tool use 모드 호출.

        return: (actions, full_messages)
        - actions: 이 응답 turn에서 호출된 tool_use 액션 리스트
                   [{"id": ..., "name": ..., "input": {...}}, ...]
        - full_messages: 다음 round-trip(tool_result 회신)에 그대로 사용할 수
                         있도록 user 메시지 + assistant 응답까지 포함된 리스트.
                         호출자가 tool_result를 추가해 재호출 가능.

        호환성: messages 미지정 시 user_prompt만으로 단일 user 메시지 생성.
        """
        client = self._ensure()
        if client is None:
            return [], []
        if messages is None:
            # 이미지가 있으면 image block + text block 함께
            if image_b64:
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                }]
            else:
                messages = [{"role": "user", "content": user_prompt}]
        # tools 배열 끝에 cache_control 한 번만 — 직전 모든 tool definition이
        # 캐시 prefix에 포함됨. system 블록까지 합쳐 ~수k 토큰이 캐시 hit.
        cached_tools = list(tools)
        if cached_tools:
            last = dict(cached_tools[-1])
            last["cache_control"] = {"type": "ephemeral"}
            cached_tools[-1] = last
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=cached_tools,
                messages=messages,
            )
            _usage.record(
                getattr(response, "usage", None), had_image=bool(image_b64),
            )
            actions: list[dict] = []
            assistant_blocks: list[dict] = []
            for block in response.content:
                btype = getattr(block, "type", None)
                if btype == "tool_use":
                    actions.append({
                        "id": getattr(block, "id", None),
                        "name": block.name,
                        "input": block.input,
                    })
                    assistant_blocks.append({
                        "type": "tool_use",
                        "id": getattr(block, "id", None),
                        "name": block.name,
                        "input": block.input,
                    })
                elif btype == "text":
                    assistant_blocks.append({
                        "type": "text", "text": block.text,
                    })
            full_messages = list(messages) + [
                {"role": "assistant", "content": assistant_blocks},
            ]
            return actions, full_messages
        except Exception as e:
            # 오프라인 감지 — APIConnectionError/APITimeoutError. hybrid 모드에서
            # 호출자가 로컬 fallback 가능하도록 raise.
            if raise_offline:
                try:
                    from anthropic import (  # type: ignore[import-not-found]
                        APIConnectionError, APITimeoutError,
                    )
                    if isinstance(e, (APIConnectionError, APITimeoutError)):
                        raise
                except ImportError:
                    pass
            log.warning(f"Claude tool 호출 실패: {e}")
            return [], []


class _HybridClient:
    """Claude 평소 + 오프라인 감지 시 로컬 LLM fallback.

    매 호출 Claude 먼저 시도. APIConnectionError/APITimeoutError 발생 시
    로컬로 자동 전환 + offline 캐시 60s (그 동안 Claude 재시도 X, 로컬 직행).
    캐시 만료 후 다음 호출에서 Claude 재시도. 성공하면 캐시 reset.

    로컬 모델은 lazy load — fallback 첫 호출 시 1회만 메모리 적재.
    평소엔 RAM 부담 0.
    """

    _OFFLINE_TTL_SEC = 60.0

    def __init__(self) -> None:
        self._claude = _ClaudeClient()
        self._local: Any = None
        self._offline_until = 0.0

    def _get_local(self) -> Any:
        if self._local is None:
            from src.brain.local_llm import get_client as _gl
            self._local = _gl()
        return self._local

    def _try_claude_first(self) -> bool:
        return time.time() >= self._offline_until

    def _mark_offline(self, reason: str) -> None:
        self._offline_until = time.time() + self._OFFLINE_TTL_SEC
        log.warning(
            f"오프라인 감지 → 로컬 LLM fallback ({self._OFFLINE_TTL_SEC:.0f}s): "
            f"{reason}"
        )

    def generate(
        self,
        user_prompt: str,
        *,
        model: str = CLAUDE_MODEL,
        max_tokens: int = 200,
        system: str = SYSTEM_PROMPT,
    ) -> str:
        if self._try_claude_first():
            try:
                client = self._claude._ensure()
                if client is None:
                    return self._get_local().generate(
                        user_prompt, max_tokens=max_tokens, system=system,
                    )
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=[{
                        "type": "text", "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }],
                    messages=[{"role": "user", "content": user_prompt}],
                )
                _usage.record(getattr(response, "usage", None))
                text = "".join(
                    b.text for b in response.content if b.type == "text"  # type: ignore[attr-defined]
                )
                self._offline_until = 0.0
                return text.strip()
            except Exception as e:
                try:
                    from anthropic import (  # type: ignore[import-not-found]
                        APIConnectionError, APITimeoutError,
                    )
                    if isinstance(e, (APIConnectionError, APITimeoutError)):
                        self._mark_offline(str(e))
                    else:
                        log.warning(f"Claude generate 실패: {e}")
                        return ""
                except ImportError:
                    log.warning(f"Claude generate 실패: {e}")
                    return ""
        # 오프라인 모드
        return self._get_local().generate(
            user_prompt, max_tokens=max_tokens, system=system,
        )

    def generate_with_tools(
        self,
        user_prompt: str,
        tools: list[dict],
        *,
        model: str = CLAUDE_MODEL,
        max_tokens: int = 300,
        system: str = SYSTEM_PROMPT,
        messages: list[dict] | None = None,
        image_b64: str | None = None,
    ) -> tuple[list[dict], list[dict]]:
        if self._try_claude_first():
            try:
                result = self._claude.generate_with_tools(
                    user_prompt, tools,
                    model=model, max_tokens=max_tokens, system=system,
                    messages=messages, image_b64=image_b64,
                    raise_offline=True,
                )
                # 성공 (응답 받음, 빈 응답이라도 OK)
                self._offline_until = 0.0
                return result
            except Exception as e:
                # raise_offline=True라 connection error만 여기 도달
                self._mark_offline(str(e))
        # 오프라인 모드 — 로컬 LLM. image는 silently ignored.
        return self._get_local().generate_with_tools(
            user_prompt, tools,
            max_tokens=max_tokens, system=system,
            messages=messages, image_b64=image_b64,
        )


# 백엔드 선택 — env LLM_BACKEND=local|claude|hybrid.
# 모두 generate / generate_with_tools 동일 인터페이스라 swap 가능.
if LLM_BACKEND == "local":
    from src.brain.local_llm import get_client as _get_local_client
    _client = _get_local_client()  # type: ignore[assignment]
    log.info(f"LLM backend: local (Qwen GGUF)")
elif LLM_BACKEND == "hybrid":
    _client = _HybridClient()  # type: ignore[assignment]
    log.info(f"LLM backend: hybrid (Claude + 오프라인 시 로컬 fallback)")
else:
    _client = _ClaudeClient()
    log.info(f"LLM backend: claude (Anthropic API)")


def generate_situational(
    event_kind: str,
    *,
    user_name: str | None = None,
    extra: dict[str, Any] | None = None,
    recent_dialog: list[dict] | None = None,
    max_tokens: int = 80,
) -> str:
    """이벤트 종류 + 현재 컨텍스트 → 자연스러운 1~2문장 한국어.

    API 키 없으면 빈 문자열 — caller가 fallback 풀 사용.
    event_kind 예: "reappear", "wave", "hands_up", "chitchat", "gaze_at_me".
    extra: 이벤트 고유 데이터 (e.g. {"absence_sec": 120}).
    recent_dialog: 최근 대화 (memory.recent_conversation 결과 형식).
    """
    from src.brain.time_of_day import period_ko
    from src.config import ANTHROPIC_API_KEY
    if not ANTHROPIC_API_KEY:
        return ""

    now = datetime.now()
    parts = [
        f"방금 일어난 일: {event_kind}",
        f"시각: {now.strftime('%H:%M')} ({period_ko(now)})",
    ]
    if user_name:
        parts.append(f"사용자 이름: {user_name}")
    if extra:
        parts.append(f"추가 정보: {json.dumps(extra, ensure_ascii=False)}")
    if recent_dialog:
        lines = []
        for r in recent_dialog[-6:]:
            who = "나" if r["speaker"] == "robot" else "사용자"
            lines.append(f"  {who}: {r['text']}")
        parts.append("최근 대화 (괄호는 비언어 행동/이벤트):\n" + "\n".join(lines))
    parts.append(
        "\n이 상황에 자연스럽게 한 마디만. 한 문장 (가끔 두 문장 OK). "
        "이전에 한 말 절대 반복 X. 의미 없는 추임새도 OK. "
        "캐릭터: 조용하고 사려 깊은 작은 로봇, 한국어 반말, 이모지 X."
    )
    prompt = "\n".join(parts)
    text = _client.generate(prompt, max_tokens=max_tokens)
    # API 키 검사는 이미 위에서 했으므로 [mock] prefix는 도달 불가.
    # 따옴표 정리만.
    text = text.strip().strip('"').strip("'").strip()
    return text


def generate_proactive_message(trigger_kind: str, context: dict[str, Any]) -> str:
    """능동 멘트 생성."""
    now = datetime.now()
    prompt = f"""현재 상황:
- 시각: {now.strftime("%Y-%m-%d %H:%M (%A)")}
- 트리거: {trigger_kind}
- 컨텍스트: {json.dumps(context, ensure_ascii=False)}

이 상황에서 사용자에게 자연스럽게 건넬 한두 문장을 작성해주세요.
잔소리가 아닌 친근한 동반자 톤으로."""
    return _client.generate(prompt, max_tokens=120)


def respond_to_user(user_text: str, history: list[dict] | None = None) -> str:
    """사용자 발화에 응답."""
    history = history or []
    convo = "\n".join(f"{h['role']}: {h['content']}" for h in history[-6:])
    prompt = f"""최근 대화:
{convo}

사용자: {user_text}

위 사용자 발화에 응답해주세요. 1~2문장."""
    return _client.generate(prompt, max_tokens=200)


def extract_schedule(transcript: str) -> list[dict]:
    """발화에서 일정/약속 JSON 추출. 없으면 빈 리스트."""
    prompt = f"""다음 발화에서 일정/약속/할일이 언급되면 JSON으로 추출:

발화: "{transcript}"

스키마:
{{
  "events": [
    {{"type": "meeting|deadline|reminder",
      "datetime": "YYYY-MM-DDTHH:MM" 또는 "" (정확하지 않으면),
      "description": "...",
      "confidence": 0.0~1.0}}
  ]
}}

일정 언급이 없으면 {{"events": []}}만 답하세요. JSON만 출력. 다른 설명 X."""
    raw = _client.generate(
        prompt,
        model=CLAUDE_MODEL_HEAVY,  # 정형 출력은 큰 모델
        max_tokens=500,
        system="당신은 텍스트에서 일정 정보만 추출하는 파서입니다. JSON만 출력합니다.",
    )
    if not raw or raw.startswith("[mock]"):
        return []
    try:
        # Claude가 가끔 코드 펜스 붙임 — 제거
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.rsplit("```", 1)[0]
        data = json.loads(cleaned)
        return data.get("events", [])
    except json.JSONDecodeError as e:
        log.warning(f"일정 JSON 파싱 실패: {e}; raw={raw[:200]}")
        return []
