"""Tests for scenarios.py: validate_corpus, select_scenarios, integration, and CLI dry-run."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from hinglish_bench.scenarios import (
    EXPECTED_PER_DOMAIN,
    EXPECTED_TOTAL,
    load_scenarios_dir,
    select_scenarios,
    validate_corpus,
)
from hinglish_bench.schemas import ExpectedToolCall, ForbiddenRule, Scenario

TOOL_NAMES = {
    "lookup_order",
    "initiate_refund",
    "check_emi_status",
    "reschedule_emi",
    "track_delivery",
    "escalate_to_human",
}


def _scenario(
    sid: str,
    domain: str,
    tool_names: list[str] | None = None,
    precedence: list[tuple[str, str]] | None = None,
    forbidden: list[str] | None = None,
) -> Scenario:
    tools = tool_names or ["lookup_order"]
    return Scenario(
        id=sid,
        domain=domain,
        caller_goal="goal",
        hidden_facts=[],
        expected_tool_calls=[ExpectedToolCall(name=t, args={}) for t in tools],
        precedence=precedence or [],
        forbidden=[ForbiddenRule(name=f, max_count=1) for f in (forbidden or [])],
        success_state={},
        max_turns=12,
    )


def _valid_corpus() -> list[Scenario]:
    """30 scenarios: 10 per domain."""
    corpus = []
    for i in range(10):
        corpus.append(_scenario(f"refund_{i:02d}", "refund"))
        corpus.append(_scenario(f"emi_{i:02d}", "emi_reminder"))
        corpus.append(_scenario(f"delivery_{i:02d}", "delivery"))
    return corpus


# ------------------------------------------------------------------ #
# validate_corpus — happy path                                         #
# ------------------------------------------------------------------ #


def test_validate_corpus_valid() -> None:
    validate_corpus(_valid_corpus(), TOOL_NAMES)  # must not raise


# ------------------------------------------------------------------ #
# validate_corpus — error cases                                        #
# ------------------------------------------------------------------ #


def test_validate_corpus_wrong_total() -> None:
    short = _valid_corpus()[:29]
    with pytest.raises(ValueError, match="expected 30 scenarios, found 29"):
        validate_corpus(short, TOOL_NAMES)


def test_validate_corpus_wrong_domain_count() -> None:
    corpus = _valid_corpus()
    # Remove one delivery, add one refund to keep total=30 but unbalance domains
    corpus = [s for s in corpus if s.id != "delivery_00"]
    corpus.append(_scenario("refund_extra", "refund"))
    with pytest.raises(ValueError, match="domain 'delivery'"):
        validate_corpus(corpus, TOOL_NAMES)


def test_validate_corpus_duplicate_id() -> None:
    corpus = _valid_corpus()
    corpus[1] = _scenario("refund_00", "emi_reminder")  # duplicate of refund_00
    with pytest.raises(ValueError, match="duplicate scenario id"):
        validate_corpus(corpus, TOOL_NAMES)


def test_validate_corpus_unknown_tool_in_expected() -> None:
    corpus = _valid_corpus()
    corpus[0] = _scenario("refund_00", "refund", tool_names=["no_such_tool"])
    with pytest.raises(ValueError, match="unknown tool in expected_tool_calls"):
        validate_corpus(corpus, TOOL_NAMES)


def test_validate_corpus_unknown_tool_in_precedence() -> None:
    corpus = _valid_corpus()
    corpus[0] = _scenario(
        "refund_00",
        "refund",
        tool_names=["lookup_order"],
        precedence=[("lookup_order", "ghost_tool")],
    )
    with pytest.raises(ValueError, match="unknown tool in precedence"):
        validate_corpus(corpus, TOOL_NAMES)


def test_validate_corpus_precedence_tool_not_in_expected() -> None:
    corpus = _valid_corpus()
    corpus[0] = _scenario(
        "refund_00",
        "refund",
        tool_names=["lookup_order"],
        precedence=[("lookup_order", "initiate_refund")],  # initiate_refund not in expected
    )
    with pytest.raises(
        ValueError, match="precedence tool 'initiate_refund' not in expected_tool_calls"
    ):
        validate_corpus(corpus, TOOL_NAMES)


def test_validate_corpus_unknown_tool_in_forbidden() -> None:
    corpus = _valid_corpus()
    corpus[0] = _scenario("refund_00", "refund", forbidden=["no_such_tool"])
    with pytest.raises(ValueError, match="unknown tool in forbidden"):
        validate_corpus(corpus, TOOL_NAMES)


def test_validate_corpus_collects_multiple_errors() -> None:
    """A corpus with multiple distinct errors raises a single ValueError listing all."""
    corpus = _valid_corpus()[:28]  # only 28 total
    corpus[0] = _scenario("refund_00", "refund", tool_names=["ghost"])  # unknown tool
    with pytest.raises(ValueError) as exc_info:
        validate_corpus(corpus, TOOL_NAMES)
    msg = str(exc_info.value)
    assert "expected 30 scenarios" in msg
    assert "unknown tool in expected_tool_calls" in msg


# ------------------------------------------------------------------ #
# select_scenarios                                                      #
# ------------------------------------------------------------------ #


def test_select_scenarios_count() -> None:
    corpus = _valid_corpus()
    selected = select_scenarios(corpus, 9)
    assert len(selected) == 9


def test_select_scenarios_stratified() -> None:
    corpus = _valid_corpus()
    selected = select_scenarios(corpus, 9)
    domains = [s.domain for s in selected]
    assert domains.count("delivery") == 3
    assert domains.count("emi_reminder") == 3
    assert domains.count("refund") == 3


def test_select_scenarios_deterministic() -> None:
    corpus = _valid_corpus()
    a = select_scenarios(corpus, 10)
    b = select_scenarios(corpus, 10)
    assert [s.id for s in a] == [s.id for s in b]


def test_select_scenarios_n_ge_total() -> None:
    corpus = _valid_corpus()
    selected = select_scenarios(corpus, 30)
    assert len(selected) == 30


def test_select_scenarios_n_exceeds_total() -> None:
    corpus = _valid_corpus()
    selected = select_scenarios(corpus, 999)
    assert len(selected) == 30


# ------------------------------------------------------------------ #
# Integration: actual 30 YAML files                                    #
# ------------------------------------------------------------------ #

SCENARIOS_DIR = Path(__file__).resolve().parents[1] / "scenarios"


@pytest.mark.skipif(
    not SCENARIOS_DIR.exists(),
    reason="scenarios/ directory not found",
)
def test_corpus_counts() -> None:
    from collections import Counter

    scenarios = load_scenarios_dir(SCENARIOS_DIR)
    assert len(scenarios) == EXPECTED_TOTAL, f"expected {EXPECTED_TOTAL}, got {len(scenarios)}"

    counts = Counter(s.domain for s in scenarios)
    for domain in ("refund", "emi_reminder", "delivery"):
        assert counts[domain] == EXPECTED_PER_DOMAIN, (
            f"domain '{domain}': expected {EXPECTED_PER_DOMAIN}, got {counts[domain]}"
        )


@pytest.mark.skipif(
    not SCENARIOS_DIR.exists(),
    reason="scenarios/ directory not found",
)
def test_corpus_validates_against_tool_names() -> None:
    scenarios = load_scenarios_dir(SCENARIOS_DIR)
    validate_corpus(scenarios, TOOL_NAMES)  # must not raise


@pytest.mark.skipif(
    not SCENARIOS_DIR.exists(),
    reason="scenarios/ directory not found",
)
def test_select_scenarios_dry_run_determinism() -> None:
    """Running select_scenarios twice on the real corpus returns identical order."""
    scenarios = load_scenarios_dir(SCENARIOS_DIR)
    a = select_scenarios(scenarios, 10)
    b = select_scenarios(scenarios, 10)
    assert [s.id for s in a] == [s.id for s in b]


# ------------------------------------------------------------------ #
# CLI dry-run (no API calls)                                           #
# ------------------------------------------------------------------ #


@pytest.mark.skipif(
    not SCENARIOS_DIR.exists(),
    reason="scenarios/ directory not found",
)
def test_cli_run_dry_run_exits_zero() -> None:
    """hcb run --dry-run must succeed without any API calls."""
    from hinglish_bench.cli import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--scenarios-dir",
            str(SCENARIOS_DIR),
            "--n-scenarios",
            "6",
            "--personas",
            "english,hinglish",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "DRY RUN" in result.output
    assert "Total conversations:" in result.output


@pytest.mark.skipif(
    not SCENARIOS_DIR.exists(),
    reason="scenarios/ directory not found",
)
def test_cli_run_dry_run_deterministic() -> None:
    """hcb run --dry-run twice lists the same scenarios in the same order."""
    from hinglish_bench.cli import app

    runner = CliRunner()

    def _run() -> list[str]:
        result = runner.invoke(
            app,
            [
                "run",
                "--scenarios-dir",
                str(SCENARIOS_DIR),
                "--n-scenarios",
                "9",
                "--personas",
                "english",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        return [line for line in result.output.splitlines() if line.strip().startswith("  ")]

    assert _run() == _run()
