"""Unit tests for task_completion, tool_correctness, and language_fit graders."""

from pathlib import Path

import yaml

from hinglish_bench.graders.language_fit import compute_agreement, detect_script
from hinglish_bench.graders.task_completion import grade_task_completion
from hinglish_bench.graders.tool_correctness import grade_tool_correctness
from hinglish_bench.schemas import (
    ExpectedToolCall,
    ForbiddenRule,
    RunRecord,
    Scenario,
    ToolCall,
    Turn,
)

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _load_scenario(name: str) -> Scenario:
    p = Path(__file__).resolve().parents[1] / "scenarios" / name
    return Scenario.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))


def _record_with_calls(*tool_calls_per_turn: list[ToolCall]) -> RunRecord:
    """Build a minimal RunRecord whose agent turns carry the given tool-call lists."""
    turns: list[Turn] = []
    for calls in tool_calls_per_turn:
        turns.append(Turn(speaker="caller", text="hi"))
        turns.append(Turn(speaker="agent", text="ok", tool_calls=calls))
    return RunRecord(
        scenario_id="test",
        persona_id="english",
        model="mock",
        run_index=0,
        turns=turns,
    )


# ------------------------------------------------------------------ #
# task_completion                                                       #
# ------------------------------------------------------------------ #


def test_grade_task_completion_pass() -> None:
    scenario = _load_scenario("refund_torn_kurta.yaml")
    record = RunRecord(
        scenario_id=scenario.id,
        persona_id="english",
        model="mock",
        run_index=0,
        final_db={"refunds": [{"order_id": "ORD-88412", "amount": 1499, "status": "initiated"}]},
    )
    result = grade_task_completion(record, scenario)
    assert result.passed is True


def test_grade_task_completion_fail_wrong_amount() -> None:
    scenario = _load_scenario("refund_torn_kurta.yaml")
    record = RunRecord(
        scenario_id=scenario.id,
        persona_id="english",
        model="mock",
        run_index=0,
        final_db={"refunds": [{"order_id": "ORD-88412", "amount": 500, "status": "initiated"}]},
    )
    result = grade_task_completion(record, scenario)
    assert result.passed is False


# ------------------------------------------------------------------ #
# tool_correctness                                                      #
# ------------------------------------------------------------------ #


def _mini_scenario(
    expected: list[ExpectedToolCall],
    precedence: list[tuple[str, str]] | None = None,
    forbidden: list[ForbiddenRule] | None = None,
) -> Scenario:
    return Scenario(
        id="test",
        domain="refund",
        caller_goal="test",
        expected_tool_calls=expected,
        precedence=precedence or [],
        forbidden=forbidden or [],
        success_state={},
    )


def test_all_satisfied() -> None:
    scenario = _mini_scenario(
        [
            ExpectedToolCall(name="lookup_order", args={"order_id": "ORD-1"}),
            ExpectedToolCall(name="initiate_refund", args={"order_id": "ORD-1"}),
        ]
    )
    record = _record_with_calls(
        [
            ToolCall(name="lookup_order", args={"order_id": "ORD-1"}),
            ToolCall(name="initiate_refund", args={"order_id": "ORD-1", "amount": 100}),
        ]
    )
    result = grade_tool_correctness(record, scenario)
    assert result.passed is True
    assert result.missing_required == []
    assert result.wrong_args == []


def test_missing_required_tool() -> None:
    scenario = _mini_scenario([ExpectedToolCall(name="lookup_order", args={"order_id": "ORD-1"})])
    record = _record_with_calls([ToolCall(name="initiate_refund", args={"order_id": "ORD-1"})])
    result = grade_tool_correctness(record, scenario)
    assert "lookup_order" in result.missing_required
    assert result.passed is False


def test_wrong_args_failure() -> None:
    """Tool name found but neither call has matching args."""
    scenario = _mini_scenario(
        [ExpectedToolCall(name="lookup_order", args={"order_id": "ORD-CORRECT"})]
    )
    record = _record_with_calls(
        [
            ToolCall(name="lookup_order", args={"order_id": "ORD-WRONG-1"}),
            ToolCall(name="lookup_order", args={"order_id": "ORD-WRONG-2"}),
        ]
    )
    result = grade_tool_correctness(record, scenario)
    assert "lookup_order" in result.wrong_args
    assert result.passed is False


def test_wrong_args_greedy_correct_match() -> None:
    """Two lookup_order calls; only the second matches expected args — greedy still finds it."""
    scenario = _mini_scenario([ExpectedToolCall(name="lookup_order", args={"order_id": "ORD-123"})])
    record = _record_with_calls(
        [
            ToolCall(name="lookup_order", args={"order_id": "ORD-999"}),
            ToolCall(name="lookup_order", args={"order_id": "ORD-123"}),
        ]
    )
    result = grade_tool_correctness(record, scenario)
    assert result.passed is True
    assert result.wrong_args == []


def test_precedence_violation() -> None:
    """b is called before a — precedence (a, b) is violated."""
    scenario = _mini_scenario(
        [
            ExpectedToolCall(name="lookup_order"),
            ExpectedToolCall(name="initiate_refund"),
        ],
        precedence=[("lookup_order", "initiate_refund")],
    )
    record = _record_with_calls(
        [
            ToolCall(name="initiate_refund", args={"order_id": "ORD-1"}),
            ToolCall(name="lookup_order", args={"order_id": "ORD-1"}),
        ]
    )
    result = grade_tool_correctness(record, scenario)
    assert ("lookup_order", "initiate_refund") in result.precedence_violations
    assert result.passed is False


def test_forbidden_exceeded() -> None:
    """initiate_refund called twice when max_count=1."""
    scenario = _mini_scenario(
        [ExpectedToolCall(name="initiate_refund", args={"order_id": "ORD-1"})],
        forbidden=[ForbiddenRule(name="initiate_refund", max_count=1)],
    )
    record = _record_with_calls(
        [
            ToolCall(name="initiate_refund", args={"order_id": "ORD-1"}),
            ToolCall(name="initiate_refund", args={"order_id": "ORD-1"}),
        ]
    )
    result = grade_tool_correctness(record, scenario)
    assert "initiate_refund" in result.forbidden_violations
    assert result.passed is False


def test_forbidden_at_limit_passes() -> None:
    """initiate_refund called exactly max_count times — not a violation."""
    scenario = _mini_scenario(
        [ExpectedToolCall(name="initiate_refund", args={"order_id": "ORD-1"})],
        forbidden=[ForbiddenRule(name="initiate_refund", max_count=1)],
    )
    record = _record_with_calls([ToolCall(name="initiate_refund", args={"order_id": "ORD-1"})])
    result = grade_tool_correctness(record, scenario)
    assert result.forbidden_violations == []


# ------------------------------------------------------------------ #
# language_fit — detect_script (sync, no LLM)                         #
# ------------------------------------------------------------------ #


def test_detect_script_devanagari() -> None:
    assert detect_script("नमस्ते मेरा ऑर्डर") == "devanagari"


def test_detect_script_latin() -> None:
    assert detect_script("Hello my order is 123") == "latin"


def test_detect_script_mixed() -> None:
    # Roughly equal amounts of Devanagari and Latin
    assert detect_script("नमस्ते Hello नमस्ते Hello नमस्ते Hello") == "mixed"


def test_detect_script_empty() -> None:
    assert detect_script("") == "empty"
    assert detect_script("123 !@#$") == "empty"  # digits and symbols don't count


# ------------------------------------------------------------------ #
# language_fit — compute_agreement                                      #
# ------------------------------------------------------------------ #


def test_compute_agreement_basic() -> None:
    judge = [3.0, 4.0, 5.0, 2.0]
    human = [3.0, 4.0, 4.0, 3.0]
    result = compute_agreement(judge, human)
    assert result["mae"] == 0.5
    assert result["within_1"] == 1.0
    assert result["pearson_r"] is not None


def test_compute_agreement_zero_variance_pearson_none() -> None:
    judge = [3.0, 3.0, 3.0]
    human = [2.0, 3.0, 4.0]
    result = compute_agreement(judge, human)
    assert result["pearson_r"] is None


def test_compute_agreement_empty_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        compute_agreement([], [])


def test_compute_agreement_length_mismatch_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        compute_agreement([3.0, 4.0], [3.0])
