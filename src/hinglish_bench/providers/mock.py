"""Scripted provider so tests and the demo run with no API keys."""

from collections.abc import Callable

from hinglish_bench.providers.base import ChatRequest, ChatResponse


class MockProvider:
    name = "mock"

    def __init__(self, responder: Callable[[ChatRequest], str]) -> None:
        self._responder = responder
        self.calls: list[ChatRequest] = []

    async def complete(self, req: ChatRequest) -> ChatResponse:
        self.calls.append(req)
        text = self._responder(req)
        # Rough token counts. Mock runs only test plumbing, not cost.
        return ChatResponse(
            text=text,
            prompt_tokens=sum(len(m.content.split()) for m in req.messages),
            completion_tokens=len(text.split()),
        )
