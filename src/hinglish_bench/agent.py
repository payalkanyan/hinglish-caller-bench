"""Agent-under-test interface, turn model, and HTTP adapter for external agents."""

from __future__ import annotations

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
    # Token counts from the agent's LLM call, if available.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    # Why the model's final call stopped — used to explain an empty text turn.
    finish_reason: str | None = None


class AgentUnderTest(Protocol):
    """Interface every agent adapter must satisfy."""

    async def respond(self, turns: list[Turn], tools: list[ToolDef]) -> AgentTurn: ...


class HttpAgent:
    """Sends the conversation to an external HTTP endpoint and parses the reply.

    The endpoint must accept POST JSON:
        {"turns": [...], "tools": [...]}
    and return:
        {"text": "...", "tool_calls": [{"name": "...", "args": {...}}, ...]}

    This adapter is provided for Stage 4+ real agent testing. Stage 2 uses
    ReferenceAgent directly.
    """

    def __init__(self, url: str, timeout: float = 60.0) -> None:
        self._url = url
        self._client = httpx.AsyncClient(timeout=timeout)

    async def respond(self, turns: list[Turn], tools: list[ToolDef]) -> AgentTurn:
        payload = {
            "turns": [t.model_dump() for t in turns],
            "tools": [t.model_dump() for t in tools],
        }
        resp = await self._client.post(self._url, json=payload)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return AgentTurn(
            text=data.get("text", ""),
            tool_calls=[ToolCall(**tc) for tc in data.get("tool_calls", [])],
        )
