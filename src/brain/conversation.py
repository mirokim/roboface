"""Claude API 통합 — 대사 생성, 일정 추출.

prompt caching으로 시스템 프롬프트 재사용해 비용 절감.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from src.config import ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_MODEL_HEAVY
from src.utils.logger import get_logger

log = get_logger("conversation")


SYSTEM_PROMPT = """당신은 사용자 책상 위에 있는 작은 캐릭터 로봇입니다.
이름: Roboface (또는 사용자가 정한 이름).

성격:
- 조용하고 사려깊음. 잔소리꾼이 아님.
- 말 거는 빈도 < 침묵 시간. 짧고 자연스러운 한두 문장만.
- 사용자를 관찰하지만 감시는 아님. 가벼운 동반자.
- 한국어로 친근한 반말 또는 가벼운 존댓말 (사용자 선호 학습).

규칙:
- 답변은 1~2문장. 절대 길게 X.
- 이모지 사용 X (음성 출력이 기본이라).
- 모를 땐 모른다고 함. 추측 X.
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
    ) -> list[dict]:
        """tool use 모드 호출. 결과: [{"name": ..., "input": {...}}, ...]"""
        client = self._ensure()
        if client is None:
            return []
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
                tools=tools,
                messages=[{"role": "user", "content": user_prompt}],
            )
            actions: list[dict] = []
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    actions.append({
                        "name": block.name,
                        "input": block.input,
                    })
            return actions
        except Exception as e:
            log.warning(f"Claude tool 호출 실패: {e}")
            return []


_client = _ClaudeClient()


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
        f"이벤트: {event_kind}",
        f"시각: {now.strftime('%H:%M')} ({period_ko(now)})",
    ]
    if user_name:
        parts.append(f"사용자 이름: {user_name}")
    if extra:
        parts.append(f"이벤트 데이터: {json.dumps(extra, ensure_ascii=False)}")
    if recent_dialog:
        lines = []
        for r in recent_dialog[-6:]:
            who = "나" if r["speaker"] == "robot" else "사용자"
            lines.append(f"  {who}: {r['text']}")
        parts.append("최근 대화 (괄호는 비언어 행동/이벤트):\n" + "\n".join(lines))
    parts.append(
        "\n이 이벤트에 자연스럽게 한 마디. 한 문장 (가끔 두 문장). "
        "이전에 한 말 절대 반복 X. 캐릭터: 조용하고 사려 깊은 작은 로봇, "
        "한국어 반말, 이모지 X."
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
