"""Aggregate grader results and render summary.json + report.md."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from hinglish_bench.graders.language_fit import LanguageFitResult
from hinglish_bench.graders.task_completion import TaskCompletionResult, grade_task_completion
from hinglish_bench.graders.tool_correctness import ToolCorrectnessResult, grade_tool_correctness
from hinglish_bench.schemas import RunRecord, Scenario
from hinglish_bench.stats import pass_at_k, wilson_ci

# ------------------------------------------------------------------ #
# Loading                                                               #
# ------------------------------------------------------------------ #


def load_records(path: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(RunRecord.model_validate_json(line))
    return records


def load_scenarios(scenarios_dir: Path) -> dict[str, Scenario]:
    scenarios: dict[str, Scenario] = {}
    for p in sorted(scenarios_dir.glob("*.yaml")):
        s = Scenario.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))
        scenarios[s.id] = s
    return scenarios


# ------------------------------------------------------------------ #
# Per-run grades (raw experiment data stays in RunRecord)              #
# ------------------------------------------------------------------ #


@dataclass
class RunGrades:
    record: RunRecord
    task: TaskCompletionResult
    tool: ToolCorrectnessResult
    language_fit: LanguageFitResult | None = None


def grade_all(records: list[RunRecord], scenarios: dict[str, Scenario]) -> list[RunGrades]:
    """Grade every record for task_completion and tool_correctness.

    Records with infra_error are skipped — a provider failure is not an agent failure
    and must not inflate the failure rate. They are reported separately.

    language_fit is left None here; it requires an async LLM judge and is optional.
    """
    grades: list[RunGrades] = []
    for r in records:
        if r.infra_error:
            continue
        scenario = scenarios.get(r.scenario_id)
        if scenario is None:
            continue
        grades.append(
            RunGrades(
                record=r,
                task=grade_task_completion(r, scenario),
                tool=grade_tool_correctness(r, scenario),
            )
        )
    return grades


# ------------------------------------------------------------------ #
# Severity scoring for worst-transcript selection                       #
# ------------------------------------------------------------------ #


def _severity(g: RunGrades) -> tuple[int, int, int, int, int]:
    """Higher severity = worse run. Used to pick the 3 worst transcripts.

    Ordering: task_fail > forbidden > wrong_args > missing_required > precedence.
    """
    return (
        0 if g.task.passed else 1,
        len(g.tool.forbidden_violations),
        len(g.tool.wrong_args),
        len(g.tool.missing_required),
        len(g.tool.precedence_violations),
    )


# ------------------------------------------------------------------ #
# Aggregation                                                           #
# ------------------------------------------------------------------ #


@dataclass
class GroupStats:
    persona_id: str
    domain: str
    n: int
    successes: int
    tool_passes: int
    success_rate: float
    success_ci: tuple[float, float]
    tool_rate: float
    tool_ci: tuple[float, float]
    mean_language_fit: float | None
    pass_at_k_val: float | None  # averaged across scenarios in this group
    k: int


def compute_report(
    grades: list[RunGrades],
    scenarios: dict[str, Scenario],
    *,
    k: int = 3,
) -> dict[str, Any]:
    """Return a nested report dict keyed by persona_id → domain → GroupStats-like dict.

    pass@k is computed at (scenario_id × persona_id × model) level first, then
    averaged across the scenarios that fall in each (persona, domain) group.

    Also returns 'worst_transcripts': list of the 3 worst RunRecords (as dicts).
    """
    # Group grades by (persona, domain)
    by_group: dict[tuple[str, str], list[RunGrades]] = defaultdict(list)
    for g in grades:
        scenario = scenarios.get(g.record.scenario_id)
        if scenario is None:
            continue
        by_group[(g.record.persona_id, scenario.domain)].append(g)

    # pass@k at (scenario_id, persona_id, model) level
    # groups by that triple, then we'll average per (persona, domain)
    scen_persona_k: dict[tuple[str, str, str], float | None] = {}
    by_scen_persona: dict[tuple[str, str, str], list[RunGrades]] = defaultdict(list)
    for g in grades:
        key = (g.record.scenario_id, g.record.persona_id, g.record.model)
        by_scen_persona[key].append(g)
    for key, gs in by_scen_persona.items():
        n = len(gs)
        c = sum(1 for g in gs if g.task.passed)
        scen_persona_k[key] = pass_at_k(n, c, k) if n >= k else None

    result: dict[str, Any] = {"groups": {}, "worst_transcripts": []}

    for (persona_id, domain), gs in sorted(by_group.items()):
        n = len(gs)
        successes = sum(1 for g in gs if g.task.passed)
        tool_passes = sum(1 for g in gs if g.tool.passed)

        # Average pass@k across the distinct (scenario, model) combinations
        scen_keys = {(g.record.scenario_id, g.record.persona_id, g.record.model) for g in gs}
        pk_vals = [scen_persona_k[sk] for sk in scen_keys if scen_persona_k.get(sk) is not None]
        pass_k = sum(pk_vals) / len(pk_vals) if pk_vals else None

        lf_scores = [g.language_fit.score for g in gs if g.language_fit is not None]
        mean_lf = sum(lf_scores) / len(lf_scores) if lf_scores else None

        result["groups"].setdefault(persona_id, {})[domain] = {
            "n": n,
            "successes": successes,
            "tool_passes": tool_passes,
            "success_rate": successes / n if n else 0.0,
            "success_ci": list(wilson_ci(successes, n)),
            "tool_rate": tool_passes / n if n else 0.0,
            "tool_ci": list(wilson_ci(tool_passes, n)),
            "mean_language_fit": mean_lf,
            "pass_at_k": pass_k,
            "k": k,
        }

    # 3 worst transcripts
    sorted_grades = sorted(grades, key=_severity, reverse=True)
    worst = sorted_grades[:3]
    result["worst_transcripts"] = [
        {
            "scenario_id": g.record.scenario_id,
            "persona_id": g.record.persona_id,
            "model": g.record.model,
            "run_index": g.record.run_index,
            "end_reason": g.record.end_reason,
            "task_passed": g.task.passed,
            "missing_required": g.tool.missing_required,
            "wrong_args": g.tool.wrong_args,
            "forbidden_violations": g.tool.forbidden_violations,
            "precedence_violations": list(g.tool.precedence_violations),
            "turns": [{"speaker": t.speaker, "text": t.text} for t in g.record.turns[:6]],
        }
        for g in worst
    ]

    return result


# ------------------------------------------------------------------ #
# Output writers                                                        #
# ------------------------------------------------------------------ #


def write_summary_json(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def write_markdown(
    report: dict[str, Any],
    grades: list[RunGrades],
    path: Path,
    records: list[RunRecord] | None = None,
) -> None:
    lines: list[str] = []
    lines.append("# hinglish-caller-bench Report\n")

    total = len(grades)
    successes = sum(1 for g in grades if g.task.passed)
    infra_errors = sum(1 for r in (records or []) if r.infra_error)
    lines.append(f"**Runs:** {total}  **Overall success:** {successes}/{total}")
    if infra_errors:
        lines.append(f"  **Infra errors (excluded from rates):** {infra_errors}")
    lines.append("\n")

    # Table
    lines.append("## Results by persona × domain\n")
    lines.append(
        "| Persona | Domain | N | Success% (95% CI) | Tool% (95% CI) | pass@k | Lang-fit |\n"
    )
    lines.append("|---|---|---|---|---|---|---|\n")

    for persona_id, domains in sorted(report["groups"].items()):
        for domain, s in sorted(domains.items()):
            ci_lo, ci_hi = s["success_ci"]
            tci_lo, tci_hi = s["tool_ci"]
            pk = f"{s['pass_at_k']:.3f}" if s["pass_at_k"] is not None else "—"
            lf = f"{s['mean_language_fit']:.1f}" if s["mean_language_fit"] is not None else "—"
            lines.append(
                f"| {persona_id} | {domain} | {s['n']} "
                f"| {s['success_rate']:.1%} [{ci_lo:.2f}, {ci_hi:.2f}] "
                f"| {s['tool_rate']:.1%} [{tci_lo:.2f}, {tci_hi:.2f}] "
                f"| {pk} | {lf} |\n"
            )

    # Failure examples
    lines.append("\n## Failure examples (3 worst transcripts)\n")
    for i, wt in enumerate(report["worst_transcripts"], 1):
        lines.append(
            f"### {i}. `{wt['scenario_id']}` / `{wt['persona_id']}` / run {wt['run_index']}\n"
        )
        lines.append(f"**End reason:** {wt['end_reason']}  \n")
        lines.append(f"**Task completion:** {'PASS' if wt['task_passed'] else 'FAIL'}  \n")
        if wt["missing_required"]:
            lines.append(f"**Missing tools:** {', '.join(wt['missing_required'])}  \n")
        if wt["wrong_args"]:
            lines.append(f"**Wrong args:** {', '.join(wt['wrong_args'])}  \n")
        if wt["forbidden_violations"]:
            lines.append(f"**Forbidden violations:** {', '.join(wt['forbidden_violations'])}  \n")
        if wt["precedence_violations"]:
            pv = ", ".join(f"{a}→{b}" for a, b in wt["precedence_violations"])
            lines.append(f"**Precedence violations:** {pv}  \n")
        lines.append("\n")
        for turn in wt["turns"]:
            prefix = "**Caller:**" if turn["speaker"] == "caller" else "**Agent:** "
            lines.append(f"> {prefix} {turn['text']}\n>\n")
        if len(wt["turns"]) == 6:
            lines.append("> *(transcript trimmed to 6 turns)*\n")
        lines.append("\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


# ------------------------------------------------------------------ #
# Top-level entry point                                                 #
# ------------------------------------------------------------------ #


def generate(
    results_dir: Path,
    scenarios_dir: Path | None = None,
    k: int = 3,
    judge_role: Any | None = None,
    lf_sample: int | None = None,
) -> None:
    """Load runs.jsonl, grade all records, and write summary.json + report.md.

    If judge_role is provided, the language-fit LLM judge runs. lf_sample limits
    scoring to a random subset of N graded records (default: all).
    """
    import asyncio
    import random

    from hinglish_bench.graders.language_fit import grade_language_fit

    if scenarios_dir is None:
        scenarios_dir = Path("scenarios")

    records_path = results_dir / "runs.jsonl"
    records = load_records(records_path)

    if not records:
        print(f"No records found in {records_path}. Nothing to report.")
        return

    scenarios = load_scenarios(scenarios_dir)
    if not scenarios:
        print(f"No scenario YAML files found in {scenarios_dir}.")
        return

    grades = grade_all(records, scenarios)
    if not grades:
        print("No records matched known scenarios. Check scenario IDs.")
        return

    # Optional language-fit pass: run async judge (optionally on a sample).
    if judge_role is not None:
        grades_to_score = grades
        if lf_sample is not None and lf_sample < len(grades):
            grades_to_score = random.sample(grades, lf_sample)
        print(f"Running language-fit judge on {len(grades_to_score)}/{len(grades)} records…")

        async def _run_lf() -> None:
            import asyncio as _asyncio
            sem = _asyncio.Semaphore(4)

            async def _one(g: RunGrades) -> None:
                async with sem:
                    try:
                        g.language_fit = await grade_language_fit(
                            g.record, g.record.persona_id, judge_role
                        )
                    except Exception as exc:
                        print(f"  language-fit failed for {g.record.scenario_id}: {exc}")

            await _asyncio.gather(*[_one(g) for g in grades_to_score])

        asyncio.run(_run_lf())
        scored = sum(1 for g in grades if g.language_fit is not None)
        print(f"  Scored {scored}/{len(grades)} records.")

    report = compute_report(grades, scenarios, k=k)
    write_summary_json(report, results_dir / "summary.json")
    write_markdown(report, grades, results_dir / "report.md", records=records)
    print(f"Wrote {results_dir / 'summary.json'} and {results_dir / 'report.md'}.")
