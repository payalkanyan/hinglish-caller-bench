"""Token-bucket limiter and backoff helper for rate-limited free-tier APIs."""

import asyncio
import random
import time


class TokenBucket:
    """Allows `rate_per_sec` requests per second on average, with `capacity` burst."""

    def __init__(self, rate_per_sec: float, capacity: float = 2.0) -> None:
        self.rate = rate_per_sec
        self.capacity = capacity
        self._tokens = capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, cost: float = 1.0) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= cost:
                    self._tokens -= cost
                    return
                await asyncio.sleep((cost - self._tokens) / self.rate)


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 60.0) -> float:
    """Exponential backoff with full jitter. `attempt` starts at 0."""
    return random.uniform(0, min(cap, base * 2**attempt))
