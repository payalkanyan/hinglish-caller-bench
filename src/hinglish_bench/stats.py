"""Wilson 95% CI and unbiased pass@k estimator."""

from __future__ import annotations

import math
from math import comb


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion.

    Returns (lower, upper) clamped to [0, 1].
    Returns (0.0, 1.0) when n == 0.
    """
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    z2 = z * z
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0.0, center - margin), min(1.0, center + margin)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of P(at least one of k randomly sampled runs passes).

    Formula: C(c, k) / C(n, k).
    n = total runs, c = successful runs, k = sample size.

    Returns 1.0 if k == 0.
    Returns 0.0 if k > c.
    Raises ValueError if k > n or c > n.
    """
    if k == 0:
        return 1.0
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    if c > n:
        raise ValueError(f"c={c} > n={n}")
    if k > c:
        return 0.0
    return comb(c, k) / comb(n, k)
