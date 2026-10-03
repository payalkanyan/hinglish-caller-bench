"""Per-role provider configuration (caller, agent, judge)."""

import os
from pathlib import Path

from pydantic import BaseModel

from hinglish_bench.providers.base import Provider, Role
from hinglish_bench.providers.cache import CachedProvider, DiskCache
from hinglish_bench.providers.openai_compat import OpenAICompatProvider
from hinglish_bench.providers.ratelimit import TokenBucket

# Free-tier presets. Any field in RoleConfig can override them.
PRESETS: dict[str, tuple[str, str]] = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}


class ConfigError(Exception):
    pass


class RoleConfig(BaseModel):
    model: str
    provider: str | None = None  # preset name: groq | gemini | openrouter
    base_url: str | None = None  # overrides the preset
    api_key_env: str | None = None  # name of the env var holding the key; overrides the preset
    requests_per_minute: float = 30.0
    concurrency: int = 4

    def resolve(self) -> tuple[str, str]:
        preset = PRESETS.get(self.provider or "", (None, None))
        base_url = self.base_url or preset[0]
        key_env = self.api_key_env or preset[1]
        if not base_url or not key_env:
            raise ConfigError("set a known provider preset, or both base_url and api_key_env")
        return base_url, key_env


class ProviderPool:
    """Builds Roles. Roles on the same account share one rate limiter."""

    def __init__(self, cache_dir: Path | None) -> None:
        self._cache = DiskCache(cache_dir) if cache_dir else None
        self._limiters: dict[tuple[str, str], TokenBucket] = {}

    def role(self, name: str, cfg: RoleConfig) -> Role:
        base_url, key_env = cfg.resolve()
        api_key = os.environ.get(key_env)
        if not api_key:
            # Name the variable, never the value.
            raise ConfigError(f"{name}: environment variable {key_env} is not set")
        limiter = self._limiters.get((base_url, key_env))
        if limiter is None:
            limiter = TokenBucket(rate_per_sec=cfg.requests_per_minute / 60)
            self._limiters[(base_url, key_env)] = limiter
        provider: Provider = OpenAICompatProvider(
            name=cfg.provider or base_url,
            base_url=base_url,
            api_key=api_key,
            limiter=limiter,
            concurrency=cfg.concurrency,
        )
        if self._cache is not None:
            provider = CachedProvider(provider, self._cache)
        return Role(name=name, provider=provider, model=cfg.model)
