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
        None, help="Directory of scenario YAML files (default: bundled scenarios)."
    ),
    results_dir: Path = typer.Option(  # noqa: B008
        Path("results"), help="Directory to write runs.jsonl into."
    ),
    n_scenarios: int = typer.Option(10, help="Number of scenarios (stratified sample)."),  # noqa: B008
    all_scenarios: bool = typer.Option(False, "--all-scenarios", help="Use all scenarios."),  # noqa: B008
    domains_opt: str = typer.Option(  # noqa: B008
        None, "--domains", help="Comma-separated domain filter: delivery,emi_reminder,refund (default: all)."
    ),
    personas_opt: str = typer.Option(  # noqa: B008
        "all", "--personas", help="Comma-separated persona IDs or 'all'."
    ),
    runs: int = typer.Option(3, help="Runs per (scenario, persona)."),  # noqa: B008
    caller_model: str = typer.Option(  # noqa: B008
        None, help="Model ID for the caller simulator (default depends on --caller-provider)."
    ),
    agent_model: str = typer.Option(  # noqa: B008
        None, help="Model ID for the reference agent (default depends on --agent-provider)."
    ),
    provider: str = typer.Option("gemini", help="Provider preset for both roles: groq | gemini | openrouter | ollama."),  # noqa: B008
    caller_provider: str = typer.Option(  # noqa: B008
        None, "--caller-provider", help="Override provider for caller simulator (splits quota)."
    ),
    agent_provider: str = typer.Option(  # noqa: B008
        None, "--agent-provider", help="Override provider for agent (splits quota)."
    ),
    cache_dir: Path = typer.Option(Path(".cache"), help="Response cache directory."),  # noqa: B008
    concurrency: int = typer.Option(4, help="Max simultaneous conversations."),  # noqa: B008
    max_turns: int = typer.Option(  # noqa: B008
        None, "--max-turns", help="Override max turns per conversation (default: from scenario YAML)."
    ),
    agent_url: str = typer.Option(  # noqa: B008
        None, "--agent-url",
        help="HTTP endpoint to evaluate. POST {messages, tools} → {text, tool_calls}. "
             "The harness executes tool calls and keeps DB state — task completion stays gradeable.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print run plan and exit."),  # noqa: B008
) -> None:
    """Run the benchmark against the reference agent or an external HTTP endpoint."""
    if scenarios_dir is None:
        scenarios_dir = Path("scenarios")

    asyncio.run(
        _run_benchmark(
            scenarios_dir=scenarios_dir,
            results_dir=results_dir,
            n_scenarios=n_scenarios,
            all_scenarios=all_scenarios,
            domains_opt=domains_opt,
            personas_opt=personas_opt,
            runs=runs,
            caller_model=caller_model,
            agent_model=agent_model,
            provider=provider,
            caller_provider=caller_provider,
            agent_provider=agent_provider,
            cache_dir=cache_dir,
            concurrency=concurrency,
            max_turns=max_turns,
            agent_url=agent_url,
            dry_run=dry_run,
        )
    )


async def _run_benchmark(
    *,
    scenarios_dir: Path,
    results_dir: Path,
    n_scenarios: int,
    all_scenarios: bool,
    domains_opt: str | None,
    personas_opt: str,
    runs: int,
    caller_model: str,
    agent_model: str,
    provider: str,
    caller_provider: str | None,
    agent_provider: str | None,
    cache_dir: Path,
    concurrency: int,
    max_turns: int | None,
    agent_url: str | None,
    dry_run: bool,
) -> None:
    from hinglish_bench.config import ProviderPool, RoleConfig
    from hinglish_bench.examples.reference_agent import TOOL_DEFS, MockDB, ReferenceAgent
    from hinglish_bench.personas import PERSONAS
    from hinglish_bench.runner import run_batch
    from hinglish_bench.scenarios import filter_by_domains, load_scenarios_dir, select_scenarios

    # 0. Resolve per-provider model defaults
    _default_models: dict[str, str] = {
        "gemini": "models/gemini-3.5-flash-lite",
        "groq": "qwen/qwen3.8-27b",
        "openrouter": "meta-llama/llama-3.3-70b-instruct",
        "ollama": "mistral:latest",
    }
    _rpm: dict[str, float] = {"groq": 25, "gemini": 14, "openrouter": 18, "ollama": 120}

    eff_caller_provider = caller_provider or provider
    eff_agent_provider = agent_provider or provider
    if caller_model is None:
        caller_model = _default_models.get(eff_caller_provider, "models/gemini-3.5-flash-lite")
    if agent_model is None:
        agent_model = _default_models.get(eff_agent_provider, "models/gemini-3.5-flash-lite")

    # 1. Load + validate corpus first (before any provider construction)
    scenarios = load_scenarios_dir(scenarios_dir)

    # 2. Apply domain filter before selection
    if domains_opt:
        domain_list = [d.strip() for d in domains_opt.split(",") if d.strip()]
        scenarios = filter_by_domains(scenarios, domain_list)
        if not scenarios:
            typer.echo(f"No scenarios match domains: {domain_list}", err=True)
            raise typer.Exit(code=1)

    # 3. Select scenarios
    selected = scenarios if all_scenarios else select_scenarios(scenarios, n_scenarios)

    # 4. Apply max_turns override (create model copies so originals stay clean)
    if max_turns is not None:
        selected = [sc.model_copy(update={"max_turns": max_turns}) for sc in selected]

    # 5. Resolve personas
    if personas_opt.strip().lower() == "all":
        persona_list = list(PERSONAS.values())
    else:
        ids = [p.strip() for p in personas_opt.split(",") if p.strip()]
        missing = [i for i in ids if i not in PERSONAS]
        if missing:
            typer.echo(f"Unknown persona IDs: {missing}", err=True)
            raise typer.Exit(code=1)
        persona_list = [PERSONAS[i] for i in ids]

    # 5. Dry-run: print plan and exit (deterministic, no provider construction)
    total_runs = len(selected) * len(persona_list) * runs
    if dry_run:
        typer.echo("=== DRY RUN ===")
        typer.echo(f"Scenarios ({len(selected)}):")
        for sc in selected:
            typer.echo(f"  {sc.id}  [{sc.domain}]  max_turns={sc.max_turns}")
        typer.echo(f"Personas ({len(persona_list)}): {[p.id for p in persona_list]}")
        typer.echo(f"Runs per combination: {runs}")
        if agent_url:
            typer.echo(f"Caller provider: {eff_caller_provider}  model: {caller_model}")
            typer.echo(f"Agent:           HTTP endpoint → {agent_url}")
        else:
            typer.echo(f"Caller provider: {eff_caller_provider}  model: {caller_model}")
            typer.echo(f"Agent provider:  {eff_agent_provider}  model: {agent_model}")
        typer.echo(f"Concurrency:  {concurrency}")
        typer.echo(f"Total conversations: {total_runs}")
        return

    # 6. Build caller provider (always needed)
    pool = ProviderPool(cache_dir=cache_dir)
    caller_cfg = RoleConfig(
        model=caller_model,
        provider=eff_caller_provider,
        requests_per_minute=_rpm.get(eff_caller_provider, 20),
    )
    caller_role = pool.role(name="caller", cfg=caller_cfg)

    # 7. Agent factory: HTTP endpoint OR reference agent
    if agent_url is not None:
        from hinglish_bench.agent import HttpAgent, ToolExecutor

        def agent_factory():  # type: ignore[return]
            db = MockDB()
            agent = HttpAgent(url=agent_url, tool_executor=ToolExecutor(db))
            return agent, db.dump

        agent_label = f"HTTP endpoint → {agent_url}"
    else:
        agent_cfg = RoleConfig(
            model=agent_model,
            provider=eff_agent_provider,
            requests_per_minute=_rpm.get(eff_agent_provider, 20),
        )
        agent_role = pool.role(name="agent", cfg=agent_cfg)

        def agent_factory():  # type: ignore[return]
            db = MockDB()
            agent = ReferenceAgent(agent_role, db)
            return agent, db.dump

        agent_label = f"{eff_agent_provider}/{agent_model}"

    # 8. Run
    results_path = results_dir / "runs.jsonl"
    typer.echo(
        f"Running {total_runs} conversations "
        f"({len(selected)} scenarios × {len(persona_list)} personas × {runs} runs) "
        f"[caller={eff_caller_provider}/{caller_model}  agent={agent_label}] …"
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

    # 9. Summary
    n_success = sum(1 for r in records if r.end_reason == "success")
    n_infra = sum(1 for r in records if r.infra_error)
    n_total = len(records)
    typer.echo(f"Ran {n_total} conversations. Success: {n_success}/{n_total}. Infra errors: {n_infra}.")
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
        None, help="Directory of scenario YAML files (default: bundled scenarios)."
    ),
    k: int = typer.Option(3, help="k for pass@k estimator."),  # noqa: B008
    language_fit: bool = typer.Option(  # noqa: B008
        False, "--language-fit", help="Run LLM language-fit judge (requires provider API key)."
    ),
    lf_sample: int = typer.Option(  # noqa: B008
        None, "--lf-sample", help="Score only a random sample of N records for language-fit (default: all)."
    ),
    provider: str = typer.Option("gemini", help="Provider for language-fit judge."),  # noqa: B008
    judge_model: str = typer.Option(  # noqa: B008
        None, help="Model for language-fit judge (default: provider default)."
    ),
) -> None:
    """Generate summary.json and report.md from a results directory."""
    if scenarios_dir is None:
        scenarios_dir = Path("scenarios")

    from hinglish_bench.report import generate

    judge_role = None
    if language_fit:
        from hinglish_bench.config import ProviderPool, RoleConfig

        _default_models: dict[str, str] = {
            "gemini": "models/gemini-3.5-flash-lite",
            "groq": "qwen/qwen3.8-27b",
        }
        model = judge_model or _default_models.get(provider, "models/gemini-3.5-flash-lite")
        pool = ProviderPool(cache_dir=Path(".cache"))
        cfg = RoleConfig(model=model, provider=provider, requests_per_minute=14)
        judge_role = pool.role(name="judge", cfg=cfg)

    generate(results_dir, scenarios_dir, k=k, judge_role=judge_role, lf_sample=lf_sample)
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


# ------------------------------------------------------------------ #
# hcb init                                                             #
# ------------------------------------------------------------------ #


@app.command()
def init(
    output_dir: Path = typer.Argument(  # noqa: B008
        Path("."), help="Directory to scaffold (default: current directory)."
    ),
) -> None:
    """Scaffold a new evaluation project with bundled scenarios and config examples."""
    import shutil

    bundled = Path(__file__).parent / "scenarios"
    if not bundled.exists():
        typer.echo("Bundled scenarios not found. Re-install the package.", err=True)
        raise typer.Exit(code=1)

    dest = output_dir / "scenarios"
    shutil.copytree(bundled, dest, dirs_exist_ok=True)
    n = len(list(dest.glob("*.yaml")))

    env_example = output_dir / ".env.example"
    env_example.write_text(
        "GROQ_API_KEY=gsk_...\n"
        "GEMINI_API_KEY=AIza...\n",
        encoding="utf-8",
    )

    adapter_example = output_dir / "adapter_example.yaml"
    adapter_example.write_text(
        "# Optional tool-name mapping when your agent uses different tool names.\n"
        "# Uncomment and edit entries as needed.\n"
        "#\n"
        "# lookup_order: get_order_details\n"
        "# initiate_refund: process_refund\n"
        "# check_emi_status: get_loan_status\n"
        "# reschedule_emi: change_emi_date\n"
        "# track_delivery: get_shipment_status\n"
        "# escalate_to_human: transfer_to_agent\n",
        encoding="utf-8",
    )

    typer.echo(f"Initialised {output_dir}  ({n} scenarios)")
    typer.echo(f"  scenarios/          ← {n} scenario YAMLs")
    typer.echo(f"  .env.example        ← copy to .env and fill in API keys")
    typer.echo(f"  adapter_example.yaml ← optional tool-name mapping")
    typer.echo("")
    typer.echo("Next steps:")
    typer.echo("  cp .env.example .env  # fill in your API keys")
    typer.echo("  # Test with a reference agent:")
    typer.echo("  hcb run --domains delivery --n-scenarios 3 --runs 1 --dry-run")
    typer.echo("  # Test your own agent endpoint:")
    typer.echo("  hcb run --agent-url http://localhost:8000/chat --n-scenarios 3 --runs 1")


def main() -> None:
    app()
