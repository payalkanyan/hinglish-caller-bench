"""Agent-under-test interface, turn model, HTTP adapter, and tool executor."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, Field

from hinglish_bench.schemas import ToolCall, Turn


class ToolDef(BaseModel):
    """A tool the agent is allowed to call, in JSON-Schema format."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class AgentTurn(BaseModel):
    """What the agent produced in a single respond() call."""

    text: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    finish_reason: str | None = None


class AgentUnderTest(Protocol):
    """Interface every agent adapter must satisfy."""

    async def respond(self, turns: list[Turn], tools: list[ToolDef]) -> AgentTurn: ...


# ------------------------------------------------------------------ #
# Tool executor — runs tool calls against MockDB by name              #
# ------------------------------------------------------------------ #


class ToolExecutor:
    """Wraps MockDB + the 6 scenario tool functions. Dispatches by tool name.

    Used by HttpAgent so the harness (not the external agent) executes tool calls,
    keeping MockDB in sync and task-completion grading valid.
    """

    def __init__(self, db: Any) -> None:
        from hinglish_bench.examples.reference_agent import (
            check_emi_status,
            escalate_to_human,
            initiate_refund,
            lookup_order,
            reschedule_emi,
            track_delivery,
        )

        self._db = db
        self._fn: dict[str, Callable[..., Any]] = {
            "lookup_order": lookup_order,
            "initiate_refund": initiate_refund,
            "check_emi_status": check_emi_status,
            "reschedule_emi": reschedule_emi,
            "track_delivery": track_delivery,
            "escalate_to_human": escalate_to_human,
        }

    def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        fn = self._fn.get(name)
        if fn is None:
            return {"error": f"unknown tool: {name}"}
        try:
            result = fn(self._db, **args)
            return result if isinstance(result, dict) else {}
        except Exception as exc:
            return {"error": str(exc)}


# ------------------------------------------------------------------ #
# HTTP agent — calls an external endpoint; harness executes tools     #
# ------------------------------------------------------------------ #


class HttpAgent:
    """Evaluates an external HTTP agent endpoint.

    Protocol: POST {messages, tools} → {text, tool_calls}

    The harness sends TOOL_DEFS (your scenario tool definitions) to the endpoint.
    When the endpoint returns tool calls, the harness executes them against MockDB
    and feeds the results back — so task completion and tool correctness remain
    fully gradeable.

    Request body:
        {
          "messages": [{"role": "system"|"user"|"assistant"|"tool", "content": "..."},
                       ...],
          "tools": [{"name": "...", "description": "...", "parameters": {...}}, ...]
        }

    Expected response:
        {
          "text": "agent reply text",
          "tool_calls": [{"name": "lookup_order", "args": {"order_id": "ORD-123"}}],
          "prompt_tokens": 120,      # optional
          "completion_tokens": 40    # optional
        }
    """

    MAX_TOOL_ROUNDS = 6

    def __init__(
        self,
        url: str,
        tool_executor: ToolExecutor,
        timeout: float = 60.0,
    ) -> None:
        self._url = url
        self._executor = tool_executor
        self._client = httpx.AsyncClient(timeout=timeout)

    async def respond(self, turns: list[Turn], tools: list[ToolDef]) -> AgentTurn:
        from hinglish_bench.examples.reference_agent import AGENT_SYSTEM_PROMPT

        # Build initial message history (system + turns so far)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT}
        ]
        for turn in turns:
            role = "user" if turn.speaker == "caller" else "assistant"
            messages.append({"role": role, "content": turn.text})

        all_tool_calls: list[ToolCall] = []
        total_prompt = total_completion = 0
        total_latency = 0.0
        text = ""

        for _ in range(self.MAX_TOOL_ROUNDS):
            t0 = time.monotonic()
            data = await self._post_once(messages, tools)
            total_latency += time.monotonic() - t0

            text = data.get("text", "")
            raw_calls = data.get("tool_calls", [])
            total_prompt += data.get("prompt_tokens", 0)
            total_completion += data.get("completion_tokens", 0)

            if not raw_calls:
                break

            tcs = [ToolCall(name=tc["name"], args=tc.get("args", {})) for tc in raw_calls]
            all_tool_calls.extend(tcs)

            # Append assistant message with tool calls, then tool results
            messages.append({
                "role": "assistant",
                "content": text or None,
                "tool_calls": [
                    {"id": tc.name, "type": "function",
                     "function": {"name": tc.name, "arguments": json.dumps(tc.args)}}
                    for tc in tcs
                ],
            })
            for tc in tcs:
                result = self._executor.execute(tc.name, tc.args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.name,
                    "content": json.dumps(result),
                })

        return AgentTurn(
            text=text,
            tool_calls=all_tool_calls,
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            latency_s=total_latency,
        )

    async def _post_once(
        self, messages: list[dict[str, Any]], tools: list[ToolDef]
    ) -> dict[str, Any]:
        resp = await self._client.post(
            self._url,
            json={"messages": messages, "tools": [t.model_dump() for t in tools]},
        )
        resp.raise_for_status()
        return resp.json()
