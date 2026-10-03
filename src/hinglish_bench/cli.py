"""CLI entry point: hcb demo | hcb run | hcb report (stubs for run/report until Stage 3+)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
import yaml

from hinglish_bench.schemas import Scenario

app = typer.Typer(help="hinglish-caller-bench — stress-test AI customer-service agents.")


# ------------------------------------------------------------------ #
# hcb demo                                                             #
# ------------------------------------------------------------------ #


@app.command()
def demo() -> None:
    """Run 2 scenarios × 2 personas on MockProvider. No API keys needed. < 10 s."""
    asyncio.run(_run_demo())


async def _run_demo() -> None:
    import time

    from rich.console import Console
    from rich.table import Table

    from hinglish_bench.examples.reference_agent import TOOL_DEFS, MockDB, ReferenceAgent
    from hinglish_bench.personas import PERSONAS
    from hinglish_bench.providers.base import ChatRequest, ChatResponse, Role, ToolCallRaw
    from hinglish_bench.providers.mock import MockProvider
    from hinglish_bench.runner import run_one

    console = Console()
    console.print("\n[bold cyan]hcb demo[/bold cyan] — running on MockProvider\n")

    # ---- Load 2 scenarios ----
    scenario_dir = Path(__file__).resolve().parents[2] / "scenarios"
    if not scenario_dir.exists():
        # Installed wheel: scenarios live next to the src tree
        scenario_dir = Path(__file__).resolve().parent.parent.parent / "scenarios"
    paths = sorted(scenario_dir.glob("*.yaml"))[:2]
    if len(paths) < 2:
        console.print("[red]Could not find scenario YAML files.[/red]")
        sys.exit(1)

    scenarios = [
        Scenario.model_validate(yaml.safe_load(p.read_text(encoding="utf-8"))) for p in paths
    ]
    personas = [PERSONAS["english"], PERSONAS["hinglish"]]

    # ---- Demo MockProvider: scripted responses for both caller and agent roles ----
    def demo_responder(req: ChatRequest) -> str | ChatResponse:
        system = req.messages[0].content if req.messages else ""
        all_content = " ".join(m.content for m in req.messages)
        is_agent = "customer service agent" in system.lower()

        if is_agent:
            # Agent: detect IDs in conversation history and produce the right tool calls.
            if "ORD-88412" in all_content:
                return ChatResponse(
                    text="I have found your order. Initiating a refund of ₹1,499 to your UPI now.",
                    tool_calls=[
                        ToolCallRaw(name="lookup_order", args={"order_id": "ORD-88412"}),
                        ToolCallRaw(
                            name="initiate_refund",
                            args={"order_id": "ORD-88412", "amount": 1499},
                        ),
                    ],
                )
            if "LN-4471032" in all_content:
                return ChatResponse(
                    text="I have checked your loan and rescheduled the EMI due date to the 25th.",
                    tool_calls=[
                        ToolCallRaw(name="check_emi_status", args={"loan_id": "LN-4471032"}),
                        ToolCallRaw(
                            name="reschedule_emi",
                            args={"loan_id": "LN-4471032", "new_due_day": 25},
                        ),
                    ],
                )
            # Opening agent turn: ask for the relevant ID.
            # Check only user messages — the agent system prompt mentions "EMI" itself.
            user_msgs = " ".join(m.content for m in req.messages if m.role == "user")
            if "emi" in user_msgs.lower() or "loan" in user_msgs.lower():
                return "Could you please share your loan ID or account number?"
            return "Could you please share your order ID or order number?"

        else:
            # Caller: once the fact is revealed in the system prompt, mention it.
            if "order_id: ORD-88412" in system:
                return "Mera order ID ORD-88412 hai. Please refund process kar do."
            if "loan_id: LN-4471032" in system:
                return "Mera loan ID LN-4471032 hai. 25 tarikh kar dijiye please."
            # Opening caller turn. Check only the goal section so persona example_lines
            # (which may mention "refund") don't create false matches.
            goal = (
                system.split("## Your goal")[1].split("##")[0].lower()
                if "## Your goal" in system
                else ""
            )
            if "refund" in goal or "kurta" in goal:
                return "Namaste! Mera kurta torn aa gaya. Mujhe refund chahiye."
            if "emi" in goal or "loan" in goal:
                return "Hello! Mujhe apna EMI due date change karna hai."
            return "Namaste, ek issue hai meri order ke baare mein."

    mock = MockProvider(demo_responder)
    caller_role = Role(name="caller", provider=mock, model="mock")
    agent_role = Role(name="agent", provider=mock, model="mock")

    t0 = time.monotonic()
    rows = []
    for scenario in scenarios:
        for persona in personas:
            db = MockDB()
            agent = ReferenceAgent(agent_role, db)
            record = await run_one(
                scenario=scenario,
                persona=persona,
                caller_role=caller_role,
                agent=agent,
                db_getter=db.dump,
                tools=TOOL_DEFS,
                run_index=0,
                results_path=None,
            )
            rows.append(record)

    elapsed = time.monotonic() - t0

    table = Table(title="Demo results", show_lines=True)
    table.add_column("Scenario", style="cyan")
    table.add_column("Persona", style="magenta")
    table.add_column("End reason", style="green")
    table.add_column("Turns", justify="right")
    table.add_column("Tokens", justify="right")

    for r in rows:
        caller_turns = sum(1 for t in r.turns if t.speaker == "caller")
        table.add_row(
            r.scenario_id,
            r.persona_id,
            r.end_reason or "—",
            str(caller_turns),
            str(r.prompt_tokens + r.completion_tokens),
        )

    console.print(table)
    console.print(
        f"\nCompleted {len(rows)} conversations in [bold]{elapsed:.2f}s[/bold] "
        f"(MockProvider, no API calls).\n"
    )


# ------------------------------------------------------------------ #
# hcb run (stub — implemented in Stage 4)                              #
# ------------------------------------------------------------------ #


@app.command()
def run(  # noqa: B008
    agent: str = typer.Option(..., help="Python import path to an AgentUnderTest class."),
    models: list[str] = typer.Option(["mock"], help="Model IDs to test."),  # noqa: B008
    runs: int = typer.Option(3, help="Runs per (scenario, persona, model)."),
    personas: str = typer.Option("all", help="Comma-separated persona IDs or 'all'."),
    scenarios_dir: Path = typer.Option(  # noqa: B008
        Path("scenarios"), help="Directory of scenario YAML files."
    ),
) -> None:
    """Run the full benchmark. (Implemented in Stage 4.)"""
    typer.echo("hcb run is not yet implemented. Coming in Stage 4.")
    raise typer.Exit(code=0)


# ------------------------------------------------------------------ #
# hcb report (stub — implemented in Stage 3)                           #
# ------------------------------------------------------------------ #


@app.command()
def report(
    results_dir: Path = typer.Argument(  # noqa: B008
        Path("results"), help="Directory containing runs.jsonl."
    ),
    scenarios_dir: Path = typer.Option(  # noqa: B008
        Path("scenarios"), help="Directory of scenario YAML files."
    ),
    k: int = typer.Option(3, help="k for pass@k estimator."),  # noqa: B008
) -> None:
    """Generate summary.json and report.md from a results directory."""
    from hinglish_bench.report import generate

    generate(results_dir, scenarios_dir, k=k)
    typer.echo(f"Report written to {results_dir}/")


def main() -> None:
    app()
