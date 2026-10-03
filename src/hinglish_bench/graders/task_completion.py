"""Task-completion grader: does final_db match the scenario's success_state?"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hinglish_bench.schemas import RunRecord, Scenario


def _row_matches(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """True if every key in `expected` is present in `actual` with equal value."""
    return all(actual.get(k) == v for k, v in expected.items())


def check_success(db_state: dict[str, Any], scenario: Scenario) -> bool:
    """True if every expected row in success_state is found in db_state.

    Each top-level key in success_state is a table name. A list value passes if
    every expected row (partial dict) matches at least one actual row.
    """
    for table, expected_rows in scenario.success_state.items():
        actual_rows: list[dict] = db_state.get(table, [])
        if not isinstance(expected_rows, list):
            if db_state.get(table) != expected_rows:
                return False
            continue
        for expected in expected_rows:
            if not any(_row_matches(expected, actual) for actual in actual_rows):
                return False
    return True


@dataclass
class TaskCompletionResult:
    passed: bool


def grade_task_completion(record: RunRecord, scenario: Scenario) -> TaskCompletionResult:
    return TaskCompletionResult(passed=check_success(record.final_db, scenario))
