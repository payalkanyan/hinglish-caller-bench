"""Scripted provider so tests and the demo run with no API keys."""

from collections.abc import Callable

from hinglish_bench.providers.base import ChatRequest, ChatResponse


class MockProvider:
    name = "mock"

    # Responder may return a plain string OR a full ChatResponse.
    # Returning ChatResponse lets callers set tool_calls without special-casing
    # inside the agent under test.
    def __init__(self, responder: Callable[[ChatRequest], str | ChatResponse]) -> None:
        self._responder = responder
        self.calls: list[ChatRequest] = []

    async def complete(self, req: ChatRequest) -> ChatResponse:
        self.calls.append(req)
        result = self._responder(req)
        if isinstance(result, ChatResponse):
            # Fill in token counts if the responder left them as zero.
            if result.prompt_tokens == 0:
                pt = sum(len(m.content.split()) for m in req.messages)
                result = result.model_copy(update={"prompt_tokens": pt})
            return result
        # Plain string result — wrap it.
        text: str = result
        return ChatResponse(
            text=text,
            prompt_tokens=sum(len(m.content.split()) for m in req.messages),
            completion_tokens=len(text.split()),
        )
