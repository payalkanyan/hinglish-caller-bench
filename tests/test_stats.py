"""Unit tests for Wilson CI and pass@k against hand-computed values."""

import math

import pytest

from hinglish_bench.stats import pass_at_k, wilson_ci

# ------------------------------------------------------------------ #
# Wilson CI                                                            #
# ------------------------------------------------------------------ #
# Hand-computed for n=10, c=7:
#   p = 0.7, z = 1.96, z² = 3.8416
#   center = (0.7 + 3.8416/20) / (1 + 3.8416/10) = 0.89208 / 1.38416 ≈ 0.64449
#   margin = 1.96 * sqrt(0.7*0.3/10 + 3.8416/400) / 1.38416
#          = 1.96 * sqrt(0.030604) / 1.38416 ≈ 0.24773
#   lower ≈ 0.3968, upper ≈ 0.8922


def test_wilson_ci_n10_c7() -> None:
    lo, hi = wilson_ci(7, 10)
    assert abs(lo - 0.3968) < 0.001, f"lower {lo:.4f} not close to 0.3968"
    assert abs(hi - 0.8922) < 0.001, f"upper {hi:.4f} not close to 0.8922"


def test_wilson_ci_zero_n() -> None:
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_wilson_ci_all_success() -> None:
    lo, hi = wilson_ci(10, 10)
    assert lo > 0.7, "all-success lower bound should be above 0.7"
    assert hi == 1.0


def test_wilson_ci_no_success() -> None:
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0
    assert hi < 0.3  # upper bound well below 30%


def test_wilson_ci_result_in_unit_interval() -> None:
    for c in range(0, 11):
        lo, hi = wilson_ci(c, 10)
        assert 0.0 <= lo <= hi <= 1.0


# ------------------------------------------------------------------ #
# pass@k                                                               #
# ------------------------------------------------------------------ #
# Hand-computed:
#   C(2,2)/C(3,2) = 1/3,  C(3,2)/C(3,2) = 1,  C(3,3)/C(5,3) = 1/10


def test_pass_at_k_3_2_2() -> None:
    assert math.isclose(pass_at_k(3, 2, 2), 1 / 3, rel_tol=1e-9)


def test_pass_at_k_3_3_2() -> None:
    assert pass_at_k(3, 3, 2) == 1.0


def test_pass_at_k_5_3_3() -> None:
    assert math.isclose(pass_at_k(5, 3, 3), 0.1, rel_tol=1e-9)


def test_pass_at_k_zero_successes() -> None:
    assert pass_at_k(5, 0, 1) == 0.0


def test_pass_at_k_k_zero() -> None:
    assert pass_at_k(5, 3, 0) == 1.0
    assert pass_at_k(0, 0, 0) == 1.0


def test_pass_at_k_perfect() -> None:
    assert pass_at_k(3, 3, 3) == 1.0


def test_pass_at_k_k_greater_than_n_raises() -> None:
    with pytest.raises(ValueError, match="k=4 > n=3"):
        pass_at_k(3, 2, 4)


def test_pass_at_k_c_greater_than_n_raises() -> None:
    with pytest.raises(ValueError, match="c=3 > n=2"):
        pass_at_k(2, 3, 1)
