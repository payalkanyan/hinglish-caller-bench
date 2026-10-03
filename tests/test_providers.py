import asyncio
import time

import pytest

from hinglish_bench.config import ConfigError, ProviderPool, RoleConfig
from hinglish_bench.providers.base import ChatMessage, Role
from hinglish_bench.providers.mock import MockProvider
from hinglish_bench.providers.ratelimit import TokenBucket, backoff_delay


def test_mock_round_trip() -> None:
    mock = MockProvider(lambda req: req.messages[-1].content.upper())
    role = Role(name="caller", provider=mock, model="mock-model")
    resp = asyncio.run(role.chat([ChatMessage(role="user", content="kya haal hai")]))
    assert resp.text == "KYA HAAL HAI"
    assert mock.calls[0].model == "mock-model"


def test_token_bucket_paces_requests() -> None:
    bucket = TokenBucket(rate_per_sec=20, capacity=1)

    async def three_acquires() -> None:
        for _ in range(3):
            await bucket.acquire()

    start = time.monotonic()
    asyncio.run(three_acquires())
    # The first is free from the burst. The next two each wait about 0.05 s.
    assert time.monotonic() - start >= 0.09


def test_backoff_is_bounded() -> None:
    for attempt in range(20):
        assert 0 <= backoff_delay(attempt, base=1.0, cap=60.0) <= 60.0


def test_missing_key_error_names_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        ProviderPool(cache_dir=None).role("caller", RoleConfig(provider="groq", model="x"))


def test_roles_on_same_account_share_a_limiter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    pool = ProviderPool(cache_dir=None)
    caller = pool.role("caller", RoleConfig(provider="groq", model="a"))
    agent = pool.role("agent", RoleConfig(provider="groq", model="b"))
    assert caller.provider._limiter is agent.provider._limiter  # type: ignore[attr-defined]
