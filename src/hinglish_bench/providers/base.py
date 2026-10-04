"""Provider interface and the request/response types shared by all providers."""

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    # Set on an assistant message that requested tools (OpenAI wire format), so the
    # follow-up request can replay the model's own tool-call turn back to it.
    tool_calls: list[dict[str, Any]] | None = None
    # Set on a tool-result message; must match the id of the assistant's tool call.
    tool_call_id: str | None = None


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.0
    seed: int | None = None
    # Function-tool schemas in OpenAI format. None means "no tools offered".
    tools: list[dict[str, Any]] | None = None
    # Not sent to the API. It only changes the cache key, so repeated runs at
    # temperature > 0 each get their own sample.
    run_index: int = Field(default=0, ge=0)


class ToolCallRaw(BaseModel):
    """A tool call as returned by the provider API (before domain conversion)."""

    id: str = ""  # OpenAI assigns an id; not all providers do
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    # Provider-specific fields to echo back when replaying this call (e.g. Gemini 3
    # requires the `extra_content.google.thought_signature` it returned). Opaque here.
    extra: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    text: str
    # Native function-calling results. Populated by OpenAICompatProvider;
    # MockProvider responders may also return these directly.
    tool_calls: list[ToolCallRaw] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False
    # Why the model stopped: "stop", "tool_calls", "length", "content_filter", …
    # Kept so an empty text turn can be explained rather than silently dropped.
    finish_reason: str | None = None


class InfraError(Exception):
    """A provider failed after all retries. Reported separately from model failures."""


class Provider(Protocol):
    name: str

    async def complete(self, req: ChatRequest) -> ChatResponse: ...


@dataclass(frozen=True)
class Role:
    """A named use of the LLM (caller, agent or judge) bound to one provider and model."""

    name: str
    provider: Provider
    model: str

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        seed: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        run_index: int = 0,
    ) -> ChatResponse:
        req = ChatRequest(
            model=self.model,
            messages=messages,
            temperature=temperature,
            seed=seed,
            tools=tools,
            run_index=run_index,
        )
        return await self.provider.complete(req)
