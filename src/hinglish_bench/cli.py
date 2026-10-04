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
            # Tool results fed back — give the final confirmation, ending the tool loop.
            if any(m.role == "tool" for m in req.messages):
                return "All done! Is there anything else I can help with?"
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
# hcb run                                                              #
# ------------------------------------------------------------------ #


@app.command()
def run(
    scenarios_dir: Path = typer.Option(  # noqa: B008
        Path("scenarios"), help="Directory of scenario YAML files."
    ),
    results_dir: Path = typer.Option(  # noqa: B008
        Path("results"), help="Directory to write runs.jsonl into."
    ),
    n_scenarios: int = typer.Option(10, help="Number of scenarios (stratified sample)."),  # noqa: B008
    all_scenarios: bool = typer.Option(False, "--all-scenarios", help="Use all 30 scenarios."),  # noqa: B008
    personas_opt: str = typer.Option(  # noqa: B008
        "all", "--personas", help="Comma-separated persona IDs or 'all'."
    ),
    runs: int = typer.Option(3, help="Runs per (scenario, persona)."),  # noqa: B008
    caller_model: str = typer.Option(  # noqa: B008
        None, help="Model ID for the caller simulator (default depends on --provider)."
    ),
    agent_model: str = typer.Option(  # noqa: B008
        None, help="Model ID for the reference agent (default depends on --provider)."
    ),
    provider: str = typer.Option("gemini", help="Provider preset: groq | gemini | openrouter."),  # noqa: B008
    cache_dir: Path = typer.Option(Path(".cache"), help="Response cache directory."),  # noqa: B008
    concurrency: int = typer.Option(4, help="Max simultaneous conversations."),  # noqa: B008
    dry_run: bool = typer.Option(False, "--dry-run", help="Print run plan and exit."),  # noqa: B008
) -> None:
    """Run the reference-agent baseline. Requires the provider's API key in environment."""
    asyncio.run(
        _run_benchmark(
            scenarios_dir=scenarios_dir,
            results_dir=results_dir,
            n_scenarios=n_scenarios,
            all_scenarios=all_scenarios,
            personas_opt=personas_opt,
            runs=runs,
            caller_model=caller_model,
            agent_model=agent_model,
            provider=provider,
            cache_dir=cache_dir,
            concurrency=concurrency,
            dry_run=dry_run,
        )
    )


async def _run_benchmark(
    *,
    scenarios_dir: Path,
    results_dir: Path,
    n_scenarios: int,
    all_scenarios: bool,
    personas_opt: str,
    runs: int,
    caller_model: str,
    agent_model: str,
    provider: str,
    cache_dir: Path,
    concurrency: int,
    dry_run: bool,
) -> None:
    from hinglish_bench.config import ProviderPool, RoleConfig
    from hinglish_bench.examples.reference_agent import TOOL_DEFS, MockDB, ReferenceAgent
    from hinglish_bench.personas import PERSONAS
    from hinglish_bench.runner import run_batch
    from hinglish_bench.scenarios import load_scenarios_dir, select_scenarios

    # 0. Resolve per-provider model defaults
    _default_models: dict[str, str] = {
        "gemini": "models/gemini-3.5-flash-lite",
        "groq": "llama-3.3-70b-versatile",
        "openrouter": "meta-llama/llama-3.3-70b-instruct",
    }
    default_model = _default_models.get(provider, "models/gemini-3.5-flash-lite")
    if caller_model is None:
        caller_model = default_model
    if agent_model is None:
        agent_model = default_model

    # 1. Load + validate corpus first (before any provider construction)
    scenarios = load_scenarios_dir(scenarios_dir)

    # 2. Select scenarios
    selected = scenarios if all_scenarios else select_scenarios(scenarios, n_scenarios)

    # 3. Resolve personas
    if personas_opt.strip().lower() == "all":
        persona_list = list(PERSONAS.values())
    else:
        ids = [p.strip() for p in personas_opt.split(",") if p.strip()]
        missing = [i for i in ids if i not in PERSONAS]
        if missing:
            typer.echo(f"Unknown persona IDs: {missing}", err=True)
            raise typer.Exit(code=1)
        persona_list = [PERSONAS[i] for i in ids]

    # 4. Dry-run: print plan and exit (deterministic, no provider construction)
    total_runs = len(selected) * len(persona_list) * runs
    if dry_run:
        typer.echo("=== DRY RUN ===")
        typer.echo(f"Scenarios ({len(selected)}):")
        for sc in selected:
            typer.echo(f"  {sc.id}  [{sc.domain}]")
        typer.echo(f"Personas ({len(persona_list)}): {[p.id for p in persona_list]}")
        typer.echo(f"Runs per combination: {runs}")
        typer.echo(f"Provider:     {provider}")
        typer.echo(f"Caller model: {caller_model}")
        typer.echo(f"Agent model:  {agent_model}")
        typer.echo(f"Concurrency:  {concurrency}")
        typer.echo(f"Total conversations: {total_runs}")
        return

    # 5. Build provider pool + roles
    # Conservative RPM per free-tier limits: Groq 30→25, Gemini 15→14, OpenRouter 20→18
    _rpm: dict[str, float] = {"groq": 25, "gemini": 14, "openrouter": 18}
    rpm = _rpm.get(provider, 20)
    pool = ProviderPool(cache_dir=cache_dir)
    caller_cfg = RoleConfig(model=caller_model, provider=provider, requests_per_minute=rpm)
    agent_cfg = RoleConfig(model=agent_model, provider=provider, requests_per_minute=rpm)
    caller_role = pool.role(name="caller", cfg=caller_cfg)
    agent_role = pool.role(name="agent", cfg=agent_cfg)

    # 6. Agent factory: fresh MockDB per run (reference-agent baseline)
    def agent_factory():  # type: ignore[return]
        db = MockDB()
        agent = ReferenceAgent(agent_role, db)
        return agent, db.dump

    # 7. Run
    results_path = results_dir / "runs.jsonl"
    typer.echo(
        f"Running {total_runs} conversations "
        f"({len(selected)} scenarios × {len(persona_list)} personas × {runs} runs) "
        f"[reference-agent baseline] …"
    )
    records = await run_batch(
        scenarios=selected,
        personas=persona_list,
        caller_role=caller_role,
        agent_factory=agent_factory,
        tools=TOOL_DEFS,
        runs=runs,
        results_path=results_path,
        max_concurrency=concurrency,
    )

    # 8. Summary
    n_success = sum(1 for r in records if r.end_reason == "success")
    n_total = len(records)
    typer.echo(f"Ran {n_total} conversations. Success: {n_success}/{n_total}.")
    typer.echo(f"Results → {results_path}")


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


# ------------------------------------------------------------------ #
# hcb plot                                                             #
# ------------------------------------------------------------------ #


@app.command()
def plot(
    results_dir: Path = typer.Argument(  # noqa: B008
        Path("results"), help="Directory containing summary.json."
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        None, help="Output directory for PNGs (default: <results_dir>/plots/)."
    ),
) -> None:
    """Generate bar charts from results/summary.json. Requires matplotlib."""
    from hinglish_bench.plot import generate_plots

    out = output_dir if output_dir is not None else (results_dir / "plots")
    summary = results_dir / "summary.json"
    if not summary.exists():
        typer.echo(f"summary.json not found in {results_dir}. Run `hcb report` first.", err=True)
        raise typer.Exit(code=1)
    paths = generate_plots(summary, out)
    for p in paths:
        typer.echo(f"Wrote {p}")


def main() -> None:
    app()
