"""원격 LLM 백엔드 — LAN의 PC에서 도는 Ollama(OpenAI 호환 API)를 호출.

Pi의 Qwen2.5-3B보다 큰 모델(7B 등)을 PC CPU/GPU로 돌리고 로봇은 HTTP로만 씀.
PC가 꺼져 있거나 안 닿으면 conversation._RemoteClient가 로컬(llama-cpp)로 fallback.

설정 (env):
  LLM_BACKEND=remote
  LLM_REMOTE_URL=http://JH_1.local:11434/v1   (OpenAI 호환 base URL)
  LLM_REMOTE_MODEL=qwen2.5:7b
  LLM_REMOTE_VISION=0|1                        (vision 모델이면 1 — 이미지 첨부 허용)

local_llm의 메시지/도구 변환기를 그대로 재사용 (Anthropic 형식 ↔ OpenAI 형식).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from src.brain.local_llm import (
    _KOREAN_ENFORCE,
    _anthropic_messages_to_openai,
    _anthropic_tools_to_openai,
    _openai_response_to_actions,
)
from src.config import LLM_REMOTE_MODEL, LLM_REMOTE_URL, LLM_REMOTE_VISION
from src.utils.logger import get_logger

log = get_logger("remote_llm")


class RemoteUnavailable(RuntimeError):
    """PC LLM 서버에 못 닿음 — 호출부가 로컬 fallback 판단용."""


class RemoteLLMClient:
    """OpenAI 호환 chat completions 클라이언트 (urllib, 추가 의존성 X).

    generate / generate_with_tools 시그니처는 _ClaudeClient/LocalLLMClient와 동일.
    """

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or LLM_REMOTE_URL).rstrip("/")
        self.model = model or LLM_REMOTE_MODEL
        self.timeout_sec = 60.0

    def available(self) -> bool:
        return bool(self.base_url and self.model)

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:200]
            raise RuntimeError(f"HTTP {e.code}: {body}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise RemoteUnavailable(str(e)) from e

    def generate(
        self,
        user_prompt: str,
        *,
        model: str = "",
        max_tokens: int = 200,
        system: str = "",
    ) -> str:
        resp = self._post({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        })
        text = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return (text or "").strip()

    def generate_with_tools(
        self,
        user_prompt: str,
        tools: list[dict],
        *,
        model: str = "",
        max_tokens: int = 300,
        system: str = "",
        messages: list[dict] | None = None,
        image_b64: str | None = None,
    ) -> tuple[list[dict], list[dict]]:
        if messages is None:
            messages = [{"role": "user", "content": user_prompt}]
        openai_messages = _anthropic_messages_to_openai(
            messages, (system or "") + _KOREAN_ENFORCE,
        )
        # vision 모델이면 마지막 user 메시지에 이미지 첨부 (OpenAI image_url 형식)
        if image_b64 and LLM_REMOTE_VISION:
            for m in reversed(openai_messages):
                if m.get("role") == "user":
                    text = m.get("content") if isinstance(m.get("content"), str) else ""
                    m["content"] = [
                        {"type": "text", "text": text or "(사진 참고)"},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ]
                    break
        t0 = time.time()
        resp = self._post({
            "model": self.model,
            "messages": openai_messages,
            "tools": _anthropic_tools_to_openai(tools),
            "tool_choice": "auto",
            "max_tokens": max_tokens,
            "temperature": 0.5,
        })
        dt = time.time() - t0
        actions, assistant_blocks = _openai_response_to_actions(resp)
        if not actions:
            choice = (resp.get("choices") or [{}])[0]
            raw_text = ((choice.get("message", {}) or {}).get("content") or "").strip()
            log.info(f"원격 LLM 추론 {dt:.1f}s, tool 호출 X → raw='{raw_text[:60]}' (speak fallback)")
            if raw_text:
                fid = f"call_fallback_{int(time.time())}"
                actions = [{"id": fid, "name": "speak", "input": {"text": raw_text}}]
                assistant_blocks = [{"type": "tool_use", "id": fid, "name": "speak",
                                     "input": {"text": raw_text}}]
        else:
            log.info(f"원격 LLM 추론 {dt:.1f}s → actions={[a['name'] for a in actions]}")
        full_messages = list(messages) + [{"role": "assistant", "content": assistant_blocks}]
        return actions, full_messages


_client: RemoteLLMClient | None = None


def get_client() -> RemoteLLMClient:
    global _client
    if _client is None:
        _client = RemoteLLMClient()
    return _client
