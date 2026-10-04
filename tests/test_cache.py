import asyncio
import json
from pathlib import Path

from hinglish_bench.providers.base import ChatMessage, ChatRequest
from hinglish_bench.providers.cache import CachedProvider, DiskCache, cache_key
from hinglish_bench.providers.mock import MockProvider


def _req(**overrides: object) -> ChatRequest:
    fields: dict[str, object] = {
        "model": "m",
        "messages": [ChatMessage(role="user", content="namaste")],
        "temperature": 0.0,
        "seed": 1,
        "run_index": 0,
    }
    fields.update(overrides)
    return ChatRequest(**fields)  # type: ignore[arg-type]


def test_key_is_stable() -> None:
    assert cache_key(_req()) == cache_key(_req())


def test_key_changes_with_each_key_field() -> None:
    base = cache_key(_req())
    assert cache_key(_req(model="other")) != base
    assert cache_key(_req(temperature=0.7)) != base
    assert cache_key(_req(seed=2)) != base
    assert cache_key(_req(run_index=1)) != base
    assert cache_key(_req(messages=[ChatMessage(role="user", content="hi")])) != base


def test_second_identical_call_is_a_hit(tmp_path: Path) -> None:
    mock = MockProvider(lambda req: "ok")
    provider = CachedProvider(mock, DiskCache(tmp_path))

    async def two_calls() -> tuple[object, object]:
        return await provider.complete(_req()), await provider.complete(_req())

    first, second = asyncio.run(two_calls())
    assert first.text == second.text == "ok"  # type: ignore[attr-defined]
    assert not first.cached  # type: ignore[attr-defined]
    assert second.cached  # type: ignore[attr-defined]
    assert len(mock.calls) == 1  # the inner provider ran once


def test_corrupt_entry_is_a_miss(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    key = cache_key(_req())
    path = tmp_path / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert cache.get(key) is None


def test_entry_holds_only_response_fields(tmp_path: Path) -> None:
    provider = CachedProvider(MockProvider(lambda req: "ok"), DiskCache(tmp_path))
    asyncio.run(provider.complete(_req()))
    stored_files = list(tmp_path.rglob("*.json"))
    assert len(stored_files) == 1
    stored = json.loads(stored_files[0].read_text(encoding="utf-8"))
    assert set(stored) == {
        "text",
        "tool_calls",
        "prompt_tokens",
        "completion_tokens",
        "latency_s",
        "finish_reason",
    }
