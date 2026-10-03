"""Provider interface and the request/response types shared by all providers."""

from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.0
    seed: int | None = None
    # Not sent to the API. It only changes the cache key, so repeated runs at
    # temperature > 0 each get their own sample.
    run_index: int = Field(default=0, ge=0)


class ChatResponse(BaseModel):
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False


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
        run_index: int = 0,
    ) -> ChatResponse:
        req = ChatRequest(
            model=self.model,
            messages=messages,
            temperature=temperature,
            seed=seed,
            run_index=run_index,
        )
        return await self.provider.complete(req)
