"""Tool-correctness grader: required calls, args, precedence, forbidden rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hinglish_bench.schemas import RunRecord, Scenario, ToolCall


def _args_subset(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """True if every key in `expected` appears in `actual` with equal value."""
    return all(actual.get(k) == v for k, v in expected.items())


@dataclass
class ToolCorrectnessResult:
    passed: bool
    missing_required: list[str] = field(default_factory=list)
    wrong_args: list[str] = field(default_factory=list)
    precedence_violations: list[tuple[str, str]] = field(default_factory=list)
    forbidden_violations: list[str] = field(default_factory=list)


def grade_tool_correctness(record: RunRecord, scenario: Scenario) -> ToolCorrectnessResult:
    """Grade a run against the scenario's tool constraints.

    Required tools: greedy match — for each ExpectedToolCall, scan unconsumed actual calls
    for one with matching name AND subset args. Categories:
      - satisfied: matched and consumed
      - wrong_args: name found but no call had matching subset args
      - missing_required: name not found at all

    Precedence: for (a, b), first-call-idx(a) < first-call-idx(b).

    Forbidden: calls to a tool more than max_count times.
    """
    all_calls: list[ToolCall] = [
        tc for t in record.turns if t.speaker == "agent" for tc in t.tool_calls
    ]

    # ------------------------------------------------------------------ #
    # 1. Required tool matching (greedy, in order)                         #
    # ------------------------------------------------------------------ #
    consumed: set[int] = set()
    missing_required: list[str] = []
    wrong_args: list[str] = []

    for exp in scenario.expected_tool_calls:
        name_found = False
        matched = False
        for i, tc in enumerate(all_calls):
            if i in consumed:
                continue
            if tc.name == exp.name:
                name_found = True
                if _args_subset(exp.args, tc.args):
                    consumed.add(i)
                    matched = True
                    break
        if not matched:
            if name_found:
                wrong_args.append(exp.name)
            else:
                missing_required.append(exp.name)

    # ------------------------------------------------------------------ #
    # 2. Precedence                                                         #
    # ------------------------------------------------------------------ #
    first_idx: dict[str, int] = {}
    for i, tc in enumerate(all_calls):
        if tc.name not in first_idx:
            first_idx[tc.name] = i

    precedence_violations: list[tuple[str, str]] = []
    for a, b in scenario.precedence:
        idx_a = first_idx.get(a)
        idx_b = first_idx.get(b)
        if idx_b is not None:
            # b was called; a must have been called first
            if idx_a is None or idx_a >= idx_b:
                precedence_violations.append((a, b))

    # ------------------------------------------------------------------ #
    # 3. Forbidden                                                          #
    # ------------------------------------------------------------------ #
    counts: dict[str, int] = {}
    for tc in all_calls:
        counts[tc.name] = counts.get(tc.name, 0) + 1

    forbidden_violations: list[str] = [
        rule.name for rule in scenario.forbidden if counts.get(rule.name, 0) > rule.max_count
    ]

    passed = not (missing_required or wrong_args or precedence_violations or forbidden_violations)
    return ToolCorrectnessResult(
        passed=passed,
        missing_required=missing_required,
        wrong_args=wrong_args,
        precedence_violations=precedence_violations,
        forbidden_violations=forbidden_violations,
    )
