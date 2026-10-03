"""Tests for runner helpers and the conversation loop.

Covers:
- check_success / partial matching
- should_end / escalation detection
- Happy-path run_one: end_reason="success", DB matches success_state
- Wrong-tool-call: wrong order_id → end_reason != "success"
- Duplicate tool call: initiate_refund called twice → transcript captures both
- save_checkpoint / load_completed_keys round-trip
"""

import asyncio
from pathlib import Path

import yaml

from hinglish_bench.examples.reference_agent import TOOL_DEFS, MockDB, ReferenceAgent
from hinglish_bench.personas import PERSONAS
from hinglish_bench.providers.base import ChatResponse, Role, ToolCallRaw
from hinglish_bench.providers.mock import MockProvider
from hinglish_bench.runner import (
    check_success,
    load_completed_keys,
    run_one,
    save_checkpoint,
    should_end,
)
from hinglish_bench.schemas import RunRecord, Scenario, ToolCall, Turn

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _load_scenario(name: str) -> Scenario:
    p = Path(__file__).resolve().parents[1] / "scenarios" / name
    return Scenario.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))


def _refund_scenario() -> Scenario:
    return _load_scenario("refund_torn_kurta.yaml")


def _emi_scenario() -> Scenario:
    return _load_scenario("emi_due_date_shift.yaml")


# ------------------------------------------------------------------ #
# check_success                                                        #
# ------------------------------------------------------------------ #


def test_check_success_match() -> None:
    scenario = _refund_scenario()
    db = {"refunds": [{"order_id": "ORD-88412", "amount": 1499, "status": "initiated"}]}
    assert check_success(db, scenario) is True


def test_check_success_partial_row_match() -> None:
    scenario = _refund_scenario()
    # Extra field in the row is fine — it's a partial match.
    db = {
        "refunds": [{"order_id": "ORD-88412", "amount": 1499, "status": "initiated", "extra": "ok"}]
    }
    assert check_success(db, scenario) is True


def test_check_success_wrong_amount_fails() -> None:
    scenario = _refund_scenario()
    db = {"refunds": [{"order_id": "ORD-88412", "amount": 999, "status": "initiated"}]}
    assert check_success(db, scenario) is False


def test_check_success_empty_table_fails() -> None:
    assert check_success({"refunds": []}, _refund_scenario()) is False


# ------------------------------------------------------------------ #
# should_end                                                           #
# ------------------------------------------------------------------ #


def test_should_end_escalated() -> None:
    turns = [
        Turn(speaker="caller", text="help"),
        Turn(
            speaker="agent",
            text="transferring",
            tool_calls=[ToolCall(name="escalate_to_human", args={"reason": "test"})],
        ),
    ]
    assert should_end(turns) == "escalated"


def test_should_end_no_escalation() -> None:
    turns = [
        Turn(speaker="caller", text="help"),
        Turn(
            speaker="agent",
            text="looking up",
            tool_calls=[ToolCall(name="lookup_order", args={"order_id": "X"})],
        ),
    ]
    assert should_end(turns) is None


def test_should_end_empty() -> None:
    assert should_end([]) is None


# ------------------------------------------------------------------ #
# Happy-path run_one                                                   #
# ------------------------------------------------------------------ #


def _make_happy_responder(scenario_id: str):
    """Return a MockProvider responder that drives a scenario to success."""

    def responder(req):
        system = req.messages[0].content if req.messages else ""
        is_agent = "customer service agent" in system.lower()
        all_content = " ".join(m.content for m in req.messages)

        if is_agent:
            if scenario_id == "refund_torn_kurta" and "ORD-88412" in all_content:
                return ChatResponse(
                    text="Refund processed.",
                    tool_calls=[
                        ToolCallRaw(name="lookup_order", args={"order_id": "ORD-88412"}),
                        ToolCallRaw(
                            name="initiate_refund",
                            args={"order_id": "ORD-88412", "amount": 1499},
                        ),
                    ],
                )
            if scenario_id == "emi_due_date_shift" and "LN-4471032" in all_content:
                return ChatResponse(
                    text="EMI rescheduled.",
                    tool_calls=[
                        ToolCallRaw(name="check_emi_status", args={"loan_id": "LN-4471032"}),
                        ToolCallRaw(
                            name="reschedule_emi",
                            args={"loan_id": "LN-4471032", "new_due_day": 25},
                        ),
                    ],
                )
            # Ask for the ID. Exclude the system message from this check because the
            # agent system prompt mentions "EMI" — it would always match otherwise.
            user_msgs = " ".join(m.content for m in req.messages if m.role == "user")
            if "emi" in user_msgs.lower() or "loan" in user_msgs.lower():
                return "Please share your loan ID or account number."
            return "Please share your order ID or order number."
        else:
            # Caller
            if scenario_id == "refund_torn_kurta":
                if "order_id: ORD-88412" in system:
                    return "My order ID is ORD-88412."
                return "I want a refund for my torn kurta."
            else:
                if "loan_id: LN-4471032" in system:
                    return "Loan ID is LN-4471032."
                return "I need to change my EMI due date."

    return responder


def test_happy_path_refund() -> None:
    scenario = _refund_scenario()
    mock = MockProvider(_make_happy_responder("refund_torn_kurta"))
    caller_role = Role("caller", mock, "mock")
    agent_role = Role("agent", mock, "mock")
    db = MockDB()
    agent = ReferenceAgent(agent_role, db)

    record = asyncio.run(
        run_one(
            scenario=scenario,
            persona=PERSONAS["english"],
            caller_role=caller_role,
            agent=agent,
            db_getter=db.dump,
            tools=TOOL_DEFS,
            run_index=0,
        )
    )

    assert record.end_reason == "success"
    assert any(
        r["order_id"] == "ORD-88412" and r["amount"] == 1499
        for r in record.final_db.get("refunds", [])
    )
    assert record.prompt_tokens > 0
    assert record.infra_error is None


def test_happy_path_emi() -> None:
    scenario = _emi_scenario()
    mock = MockProvider(_make_happy_responder("emi_due_date_shift"))
    caller_role = Role("caller", mock, "mock")
    agent_role = Role("agent", mock, "mock")
    db = MockDB()
    agent = ReferenceAgent(agent_role, db)

    record = asyncio.run(
        run_one(
            scenario=scenario,
            persona=PERSONAS["hinglish"],
            caller_role=caller_role,
            agent=agent,
            db_getter=db.dump,
            tools=TOOL_DEFS,
            run_index=1,
        )
    )

    assert record.end_reason == "success"
    emis = record.final_db.get("emis", [])
    assert any(e.get("loan_id") == "LN-4471032" and e.get("due_day") == 25 for e in emis)


# ------------------------------------------------------------------ #
# Fix 4: wrong-tool-call test (wrong order_id)                         #
# ------------------------------------------------------------------ #


def test_wrong_order_id_does_not_succeed() -> None:
    """Agent calls initiate_refund with a wrong order_id — success_state not met."""
    scenario = _refund_scenario()

    def responder(req):
        system = req.messages[0].content if req.messages else ""
        is_agent = "customer service agent" in system.lower()
        if is_agent:
            # Always refund the WRONG order.
            return ChatResponse(
                text="Refund done.",
                tool_calls=[
                    ToolCallRaw(
                        name="initiate_refund",
                        args={"order_id": "WRONG-ID", "amount": 1499},
                    )
                ],
            )
        return "I want a refund for ORD-88412."

    mock = MockProvider(responder)
    db = MockDB()
    agent = ReferenceAgent(Role("agent", mock, "mock"), db)

    record = asyncio.run(
        run_one(
            scenario=scenario,
            persona=PERSONAS["english"],
            caller_role=Role("caller", mock, "mock"),
            agent=agent,
            db_getter=db.dump,
            tools=TOOL_DEFS,
            run_index=0,
        )
    )

    # The wrong refund is recorded in the DB — the runner captures reality faithfully.
    assert any(r["order_id"] == "WRONG-ID" for r in record.final_db.get("refunds", []))
    # But success_state expects ORD-88412, so the run should not end as "success".
    assert record.end_reason != "success"
    # Transcript has the bad tool call.
    agent_turns = [t for t in record.turns if t.speaker == "agent"]
    all_tc_names = [tc.name for t in agent_turns for tc in t.tool_calls]
    assert "initiate_refund" in all_tc_names


# ------------------------------------------------------------------ #
# Fix 4: duplicate tool call test                                      #
# ------------------------------------------------------------------ #


def test_duplicate_tool_call_captured_in_transcript() -> None:
    """Agent calls initiate_refund twice in a single turn — both calls appear in the transcript.

    This mimics a misbehaving agent that issues the same side-effect call twice in one
    response. The runner must capture both in the transcript so the Stage 3 grader can
    detect the violation using the ForbiddenRule (max_count=1 for initiate_refund).
    """
    scenario = _refund_scenario()

    def responder(req):
        system = req.messages[0].content if req.messages else ""
        is_agent = "customer service agent" in system.lower()
        if is_agent:
            # Two initiate_refund calls in a single response — the forbidden duplicate.
            return ChatResponse(
                text="I have processed the refund (twice, mistakenly).",
                tool_calls=[
                    ToolCallRaw(
                        name="initiate_refund",
                        args={"order_id": "ORD-88412", "amount": 1499},
                    ),
                    ToolCallRaw(
                        name="initiate_refund",
                        args={"order_id": "ORD-88412", "amount": 1499},
                    ),
                ],
            )
        return "I want a refund for ORD-88412."

    mock = MockProvider(responder)
    db = MockDB()
    agent = ReferenceAgent(Role("agent", mock, "mock"), db)

    record = asyncio.run(
        run_one(
            scenario=scenario,
            persona=PERSONAS["english"],
            caller_role=Role("caller", mock, "mock"),
            agent=agent,
            db_getter=db.dump,
            tools=TOOL_DEFS,
            run_index=0,
        )
    )

    refund_calls = [
        tc
        for t in record.turns
        if t.speaker == "agent"
        for tc in t.tool_calls
        if tc.name == "initiate_refund"
    ]
    # Both calls from the same turn appear in the transcript — grader can count them.
    assert len(refund_calls) == 2
    # The DB also reflects both mutations (two rows).
    assert len(record.final_db.get("refunds", [])) == 2


# ------------------------------------------------------------------ #
# Checkpoint round-trip                                               #
# ------------------------------------------------------------------ #


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    record = RunRecord(
        scenario_id="s1",
        persona_id="english",
        model="mock",
        run_index=0,
        end_reason="success",
    )
    save_checkpoint(record, path)
    save_checkpoint(record.model_copy(update={"run_index": 1}), path)

    keys = load_completed_keys(path)
    assert ("s1", "english", "mock", 0) in keys
    assert ("s1", "english", "mock", 1) in keys
    assert ("s1", "english", "mock", 2) not in keys


def test_checkpoint_skip_completed(tmp_path: Path) -> None:
    """run_batch skips runs already in the JSONL file."""
    from hinglish_bench.runner import run_batch

    scenario = _refund_scenario()
    path = tmp_path / "runs.jsonl"
    mock = MockProvider(lambda req: "done")
    caller_role = Role("caller", mock, "mock")

    def factory():
        db = MockDB()
        return ReferenceAgent(Role("agent", mock, "mock"), db), db.dump

    # Pre-populate one completed run.
    existing = RunRecord(
        scenario_id=scenario.id,
        persona_id="english",
        model="mock",
        run_index=0,
        end_reason="max_turns",
    )
    save_checkpoint(existing, path)

    records = asyncio.run(
        run_batch(
            scenarios=[scenario],
            personas=[PERSONAS["english"]],
            caller_role=caller_role,
            agent_factory=factory,
            tools=TOOL_DEFS,
            runs=2,  # 2 runs requested
            results_path=path,
        )
    )

    # Only run_index=1 should have been executed (run_index=0 was pre-loaded).
    assert all(r.run_index == 1 for r in records)
    assert len(records) == 1
