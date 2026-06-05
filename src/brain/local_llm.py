"""로컬 LLM 백엔드 — llama-cpp-python 기반 Qwen2.5-3B Instruct.

오프라인 동작 + 비용 0. Pi5 CPU (Cortex-A76 NEON)에서 5-10 tok/s.
Anthropic tool 스키마와 동일한 (actions, full_messages) 인터페이스 제공해
_ClaudeClient와 swap 가능.

vision(이미지 첨부)은 미지원 — image_b64 인자는 silently ignore.
대신 perception이 텍스트로 시각 신호 요약해 user prompt에 포함.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from src.config import BEHAVIOR
from src.utils.logger import get_logger

log = get_logger("local_llm")


# 모델 경로 — 봇 setup에서 받음. 없으면 LocalLLMClient.available()이 False.
LOCAL_MODEL_PATH = Path(
    os.getenv(
        "LOCAL_MODEL_PATH",
        str(Path(__file__).resolve().parents[2] / "models"
            / "qwen2.5-3b-instruct-q4_k_m.gguf"),
    )
)
# 컨텍스트 길이 — agent prompt + tools + history가 들어가야. 4096이면 충분.
LOCAL_N_CTX = int(os.getenv("LOCAL_N_CTX", "4096"))
# 쓰레드 수 — Pi5는 4 core. 카메라/STT랑 양보해 3 권장.
LOCAL_N_THREADS = int(os.getenv("LOCAL_N_THREADS", "3"))


def _anthropic_tools_to_openai(tools: list[dict]) -> list[dict]:
    """Anthropic tool 스키마 → OpenAI function 스키마.

    Anthropic: {"name", "description", "input_schema": {...}}
    OpenAI:    {"type":"function", "function":{"name","description","parameters":{...}}}
    """
    out: list[dict] = []
    for t in tools:
        # cache_control 같은 Anthropic-only 필드는 drop
        name = t.get("name")
        if not name:
            continue
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return out


def _flatten_content(content: Any) -> str:
    """Anthropic content blocks → 단일 텍스트.

    Anthropic은 [{"type":"text","text":"..."}, {"type":"image",...}, ...]도 받지만
    로컬 모델은 vision 없으니 text만 추출. image는 "[이미지 첨부됨]" 마커로 대체.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for blk in content:
            btype = blk.get("type") if isinstance(blk, dict) else None
            if btype == "text":
                parts.append(blk.get("text", ""))
            elif btype == "image":
                parts.append("[시각 컨텍스트: 사용자 모습 (텍스트 요약 참조)]")
            elif btype == "tool_result":
                inner = blk.get("content", "")
                if isinstance(inner, list):
                    inner = "".join(
                        b.get("text", "") for b in inner
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                parts.append(f"[도구 결과] {inner}")
        return "\n".join(p for p in parts if p)
    return str(content)


def _anthropic_messages_to_openai(messages: list[dict], system: str) -> list[dict]:
    """Anthropic messages 배열 → OpenAI messages 배열.

    system 파라미터는 messages 맨 앞에 role="system"으로 prepend.
    assistant tool_use 블록은 OpenAI tool_calls 형식으로 변환.
    user tool_result 블록은 role="tool"로 변환.
    """
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "assistant" and isinstance(content, list):
            # tool_use 블록이 섞인 assistant 메시지
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for blk in content:
                btype = blk.get("type") if isinstance(blk, dict) else None
                if btype == "text":
                    text_parts.append(blk.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": blk.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": blk.get("name", ""),
                            "arguments": json.dumps(
                                blk.get("input", {}), ensure_ascii=False
                            ),
                        },
                    })
            msg: dict = {"role": "assistant"}
            if text_parts:
                msg["content"] = "\n".join(text_parts)
            else:
                msg["content"] = ""
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
        elif role == "user" and isinstance(content, list) \
                and any(isinstance(b, dict) and b.get("type") == "tool_result"
                        for b in content):
            # tool_result는 OpenAI에선 role="tool"로 분리
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "tool_result":
                    inner = blk.get("content", "")
                    if isinstance(inner, list):
                        inner = "".join(
                            b.get("text", "") for b in inner
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    out.append({
                        "role": "tool",
                        "tool_call_id": blk.get("tool_use_id", ""),
                        "content": str(inner),
                    })
                elif isinstance(blk, dict) and blk.get("type") == "text":
                    out.append({"role": "user", "content": blk.get("text", "")})
        else:
            out.append({"role": role, "content": _flatten_content(content)})
    return out


def _openai_response_to_actions(response: dict) -> tuple[list[dict], list[dict]]:
    """llama-cpp create_chat_completion 응답 → (actions, assistant_blocks).

    actions: [{"id","name","input"}, ...] — _ClaudeClient와 동일 형식.
    assistant_blocks: Anthropic 형식의 assistant content 블록 (full_messages용).
    """
    actions: list[dict] = []
    assistant_blocks: list[dict] = []
    choice = (response.get("choices") or [{}])[0]
    msg = choice.get("message", {}) or {}
    text = msg.get("content")
    if text:
        assistant_blocks.append({"type": "text", "text": text})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {}) or {}
        name = fn.get("name")
        raw_args = fn.get("arguments", "{}")
        try:
            inp = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except json.JSONDecodeError:
            log.warning(f"tool args JSON parse 실패: {raw_args!r}")
            inp = {}
        if not name:
            continue
        tid = tc.get("id") or f"call_{len(actions)}"
        actions.append({"id": tid, "name": name, "input": inp})
        assistant_blocks.append({
            "type": "tool_use",
            "id": tid,
            "name": name,
            "input": inp,
        })
    return actions, assistant_blocks


class LocalLLMClient:
    """llama-cpp-python 기반 로컬 추론 클라이언트.

    _ClaudeClient와 동일한 generate / generate_with_tools 시그니처 제공.
    모델은 lazy load — 첫 호출 시 1회만. 이후 in-memory.

    스레드 안전성: llama-cpp 추론은 single-threaded model. _lock으로 직렬화.
    """

    def __init__(self) -> None:
        self._llm: Any = None
        self._tried = False
        self._lock = threading.Lock()

    def available(self) -> bool:
        return LOCAL_MODEL_PATH.exists()

    def _ensure(self) -> Any:
        if self._llm is not None:
            return self._llm
        if self._tried:
            return None
        self._tried = True
        if not self.available():
            log.warning(f"로컬 모델 파일 없음: {LOCAL_MODEL_PATH}")
            return None
        try:
            from llama_cpp import Llama  # type: ignore[import-not-found]
        except ImportError as e:
            log.warning(f"llama-cpp-python 미설치: {e}")
            return None
        try:
            log.info(
                f"로컬 LLM 로드 중: {LOCAL_MODEL_PATH.name} "
                f"(ctx={LOCAL_N_CTX}, threads={LOCAL_N_THREADS})"
            )
            t0 = time.time()
            self._llm = Llama(
                model_path=str(LOCAL_MODEL_PATH),
                n_ctx=LOCAL_N_CTX,
                n_threads=LOCAL_N_THREADS,
                # Qwen2.5는 chatml + native tool calling.
                # llama-cpp-python이 모델 metadata에서 자동 감지하지만
                # 명시해 안정성 확보.
                chat_format="chatml-function-calling",
                verbose=False,
            )
            log.info(f"로컬 LLM 로드 완료 ({time.time() - t0:.1f}s)")
            return self._llm
        except Exception as e:
            log.warning(f"로컬 LLM 로드 실패: {e}")
            return None

    def generate(
        self,
        user_prompt: str,
        *,
        model: str = "",  # 무시 — 단일 모델
        max_tokens: int = 200,
        system: str = "",
    ) -> str:
        llm = self._ensure()
        if llm is None:
            return ""
        try:
            with self._lock:
                resp = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
            text = (resp.get("choices") or [{}])[0].get("message", {}).get(
                "content", ""
            )
            return (text or "").strip()
        except Exception as e:
            log.warning(f"로컬 LLM generate 실패: {e}")
            return ""

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
        """tool use 모드. _ClaudeClient.generate_with_tools와 동일 인터페이스.

        image_b64는 silently ignore (Qwen2.5-3B는 vision 없음).
        """
        llm = self._ensure()
        if llm is None:
            return [], []
        if messages is None:
            messages = [{"role": "user", "content": user_prompt}]
        openai_messages = _anthropic_messages_to_openai(messages, system)
        openai_tools = _anthropic_tools_to_openai(tools)
        try:
            with self._lock:
                t0 = time.time()
                resp = llm.create_chat_completion(
                    messages=openai_messages,
                    tools=openai_tools,
                    tool_choice="auto",
                    max_tokens=max_tokens,
                    temperature=0.5,
                )
                dt = time.time() - t0
            actions, assistant_blocks = _openai_response_to_actions(resp)
            log.debug(
                f"로컬 LLM 추론 {dt:.1f}s → actions={[a['name'] for a in actions]}"
            )
            full_messages = list(messages) + [
                {"role": "assistant", "content": assistant_blocks}
            ]
            return actions, full_messages
        except Exception as e:
            log.warning(f"로컬 LLM tool 호출 실패: {e}")
            return [], []


# 모듈 싱글톤
_client: LocalLLMClient | None = None


def get_client() -> LocalLLMClient:
    global _client
    if _client is None:
        _client = LocalLLMClient()
    return _client
