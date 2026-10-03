"""Conversation loop, helper functions, checkpointing, and batch runner."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from hinglish_bench.agent import AgentTurn, AgentUnderTest, ToolDef
from hinglish_bench.graders.task_completion import check_success  # noqa: F401
from hinglish_bench.providers.base import InfraError, Role
from hinglish_bench.schemas import Persona, RunRecord, Scenario, Turn

EndReason = Literal["success", "escalated", "max_turns", "infra_error"]


def should_end(turns: list[Turn]) -> EndReason | None:
    """Return a non-max_turns end reason if the last agent turn signals it, else None.

    max_turns is checked by the loop itself (not here), so this only looks for
    escalation signals in the agent's most recent turn.
    """
    # Walk backwards to find the last agent turn.
    for turn in reversed(turns):
        if turn.speaker == "agent":
            if any(tc.name == "escalate_to_human" for tc in turn.tool_calls):
                return "escalated"
            break
    return None


def save_checkpoint(record: RunRecord, path: Path) -> None:
    """Append one JSONL line. Creates the file and parent dirs on first write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(record.model_dump_json() + "\n")


def load_completed_keys(path: Path) -> set[tuple[str, str, str, int]]:
    """Return (scenario_id, persona_id, model, run_index) for every line in the JSONL."""
    keys: set[tuple[str, str, str, int]] = set()
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
            keys.add((d["scenario_id"], d["persona_id"], d["model"], d["run_index"]))
        except (json.JSONDecodeError, KeyError):
            pass  # Corrupted line — skip but don't crash.
    return keys


# ------------------------------------------------------------------ #
# Conversation loop                                                    #
# ------------------------------------------------------------------ #


async def run_one(
    scenario: Scenario,
    persona: Persona,
    caller_role: Role,
    agent: AgentUnderTest,
    db_getter: Callable[[], dict[str, Any]],
    tools: list[ToolDef],
    run_index: int = 0,
    results_path: Path | None = None,
) -> RunRecord:
    """Run one complete caller–agent conversation and return a RunRecord.

    `db_getter` is called after each agent turn to get the current DB state.
    For ReferenceAgent, pass `agent.db.dump`; for an HTTP agent, pass a no-op.

    `results_path`: if set, the record is appended as JSONL after the run.
    """
    from hinglish_bench.simulator import CallerSimulator

    simulator = CallerSimulator(scenario, persona, caller_role, run_index)
    turns: list[Turn] = []
    total_prompt = 0
    total_completion = 0
    start = time.monotonic()
    end_reason: EndReason = "max_turns"
    infra_err: str | None = None

    try:
        for _ in range(scenario.max_turns):
            # --- Caller turn ---
            caller_turn, pt, ct = await simulator.next_turn(turns)
            turns.append(caller_turn)
            total_prompt += pt
            total_completion += ct

            # --- Agent turn ---
            agent_turn: AgentTurn = await agent.respond(turns, tools)
            turns.append(
                Turn(
                    speaker="agent",
                    text=agent_turn.text,
                    tool_calls=agent_turn.tool_calls,
                )
            )
            total_prompt += agent_turn.prompt_tokens
            total_completion += agent_turn.completion_tokens

            # --- End-condition checks ---
            db = db_getter()
            if check_success(db, scenario):
                end_reason = "success"
                break
            ec = should_end(turns)
            if ec:
                end_reason = ec
                break
    except InfraError as exc:
        end_reason = "infra_error"
        infra_err = str(exc)

    record = RunRecord(
        scenario_id=scenario.id,
        persona_id=persona.id,
        model=caller_role.model,
        run_index=run_index,
        turns=turns,
        final_db=db_getter(),
        prompt_tokens=total_prompt,
        completion_tokens=total_completion,
        latency_s=time.monotonic() - start,
        infra_error=infra_err,
        end_reason=end_reason,
    )
    if results_path:
        save_checkpoint(record, results_path)
    return record


async def run_batch(
    scenarios: list[Scenario],
    personas: list[Persona],
    caller_role: Role,
    agent_factory: Callable[[], tuple[AgentUnderTest, Callable[[], dict[str, Any]]]],
    tools: list[ToolDef],
    runs: int = 3,
    results_path: Path | None = None,
    max_concurrency: int = 1,
) -> list[RunRecord]:
    """Run all (scenario, persona, run_index) combinations, skipping completed ones.

    `agent_factory` returns a fresh (agent, db_getter) pair for each run so every
    run starts with a clean DB state.

    `max_concurrency` limits simultaneous in-flight conversations. Rate limiting is
    handled by the TokenBucket inside ProviderPool (shared per base_url/key pair).
    """
    import asyncio

    completed = load_completed_keys(results_path) if results_path else set()
    sem = asyncio.Semaphore(max_concurrency)

    async def _one(scenario: Scenario, persona: Persona, run_idx: int) -> RunRecord | None:
        key = (scenario.id, persona.id, caller_role.model, run_idx)
        if key in completed:
            return None
        async with sem:
            agent, db_getter = agent_factory()
            return await run_one(
                scenario=scenario,
                persona=persona,
                caller_role=caller_role,
                agent=agent,
                db_getter=db_getter,
                tools=tools,
                run_index=run_idx,
                results_path=results_path,
            )

    tasks = [_one(sc, pe, ri) for sc in scenarios for pe in personas for ri in range(runs)]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]
