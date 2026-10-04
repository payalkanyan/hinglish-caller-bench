"""On-disk response cache. Reruns of the same request cost nothing."""

import hashlib
import json
import os
from pathlib import Path

from hinglish_bench.providers.base import ChatRequest, ChatResponse, Provider


def cache_key(req: ChatRequest) -> str:
    # The key holds only request content, never the API key or base URL.
    payload = {
        "model": req.model,
        "messages": [m.model_dump(exclude_none=True) for m in req.messages],
        "temperature": req.temperature,
        "seed": req.seed,
        "tools": req.tools,
        "run_index": req.run_index,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        # Two-character fan-out keeps directories small.
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> ChatResponse | None:
        path = self._path(key)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None  # A corrupt entry is treated as a miss and is overwritten.
        return ChatResponse(**data, cached=True)

    def put(self, key: str, resp: ChatResponse) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        stored = {
            "text": resp.text,
            "tool_calls": [tc.model_dump() for tc in resp.tool_calls],
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "latency_s": resp.latency_s,
            "finish_reason": resp.finish_reason,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(stored, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)  # Atomic, so an interrupted write never leaves half a file.


class CachedProvider:
    """Wraps any provider. Checks the cache first and stores every fresh response."""

    def __init__(self, inner: Provider, cache: DiskCache) -> None:
        self.inner = inner
        self.cache = cache
        self.name = inner.name

    async def complete(self, req: ChatRequest) -> ChatResponse:
        key = cache_key(req)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        resp = await self.inner.complete(req)
        self.cache.put(key, resp)
        return resp
