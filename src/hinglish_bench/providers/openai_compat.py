"""Provider for any OpenAI-compatible endpoint (Groq, OpenRouter, Gemini)."""

import asyncio
import json
import time

import openai
from openai import AsyncOpenAI

from hinglish_bench.providers.base import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    InfraError,
    ToolCallRaw,
)
from hinglish_bench.providers.ratelimit import TokenBucket, backoff_delay

# Errors worth retrying. Auth and bad-request errors are not, and propagate as-is.
RETRYABLE = (
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APIConnectionError,
    openai.APITimeoutError,
)


def _as_api_message(m: ChatMessage) -> dict:
    """Serialize a message for the API in OpenAI wire format.

    An assistant message that carries tool_calls, and a tool-result message, are
    passed through with their function-calling fields. For plain text turns we
    guarantee non-empty content: Gemini (and some OpenAI-compatible endpoints)
    silently drop messages whose content is empty or whitespace, and a dropped
    trailing turn can leave the request "ending on a model turn", which Gemini
    rejects. A model that emits an empty turn is a weak agent, not an infra
    failure — so we keep the turn with a visible placeholder rather than crash the
    run. The stored RunRecord keeps the true (empty) text.
    """
    if m.tool_calls:
        # A pure tool-call turn may legitimately have empty content.
        return {"role": m.role, "content": m.content, "tool_calls": m.tool_calls}
    if m.role == "tool":
        return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content or "(empty)"}
    return {"role": m.role, "content": m.content if m.content.strip() else "(no response)"}


class OpenAICompatProvider:
    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        limiter: TokenBucket,
        concurrency: int = 4,
        max_attempts: int = 6,
    ) -> None:
        self.name = name
        # The SDK's own retries are off, so 429s go through our limiter and backoff.
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        self._limiter = limiter
        self._sem = asyncio.Semaphore(concurrency)
        self._max_attempts = max_attempts

    async def complete(self, req: ChatRequest) -> ChatResponse:
        last: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                return await self._call_once(req)
            except RETRYABLE as exc:
                last = exc
                await asyncio.sleep(backoff_delay(attempt))
        raise InfraError(f"{self.name}: {self._max_attempts} attempts failed") from last

    async def _call_once(self, req: ChatRequest) -> ChatResponse:
        kwargs: dict[str, object] = {
            "model": req.model,
            "messages": [_as_api_message(m) for m in req.messages],
            "temperature": req.temperature,
        }
        if req.seed is not None:
            kwargs["seed"] = req.seed
        if req.tools:
            kwargs["tools"] = req.tools
        async with self._sem:
            await self._limiter.acquire()
            start = time.monotonic()
            resp = await self._client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
        usage = resp.usage
        choice = resp.choices[0]
        msg = choice.message

        # Parse native function-calling tool calls if the model returned any.
        raw_calls: list[ToolCallRaw] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                # Gemini 3 returns a thought_signature under extra_content that must be
                # echoed back verbatim when the call is replayed in the next request.
                tc_extra = getattr(tc, "model_extra", None) or {}
                extra = (
                    {"extra_content": tc_extra["extra_content"]}
                    if tc_extra.get("extra_content")
                    else None
                )
                raw_calls.append(
                    ToolCallRaw(id=tc.id or "", name=tc.function.name, args=args, extra=extra)
                )

        return ChatResponse(
            text=msg.content or "",
            tool_calls=raw_calls,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_s=time.monotonic() - start,
            finish_reason=choice.finish_reason,
        )
