"""Scenario corpus loading, validation, and deterministic stratified selection."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

from hinglish_bench.schemas import Scenario

EXPECTED_TOTAL = 30
EXPECTED_PER_DOMAIN = 10


def load_scenarios_dir(scenarios_dir: Path) -> list[Scenario]:
    """Load all *.yaml files from scenarios_dir, sorted alphabetically."""
    paths = sorted(scenarios_dir.glob("*.yaml"))
    return [Scenario.model_validate(yaml.safe_load(p.read_text(encoding="utf-8"))) for p in paths]


def validate_corpus(scenarios: list[Scenario], tool_names: set[str]) -> None:
    """Validate the scenario corpus structure. Collects all errors before raising.

    Checks: total count, per-domain counts, unique IDs, tool names,
    precedence tool membership, and forbidden tool names.
    """
    errors: list[str] = []

    if len(scenarios) != EXPECTED_TOTAL:
        errors.append(f"expected {EXPECTED_TOTAL} scenarios, found {len(scenarios)}")

    domain_counts = Counter(s.domain for s in scenarios)
    for domain in ("refund", "emi_reminder", "delivery"):
        count = domain_counts.get(domain, 0)
        if count != EXPECTED_PER_DOMAIN:
            errors.append(f"domain '{domain}': expected {EXPECTED_PER_DOMAIN}, found {count}")

    ids = [s.id for s in scenarios]
    for sid, count in Counter(ids).items():
        if count > 1:
            errors.append(f"duplicate scenario id: {sid!r}")

    for s in scenarios:
        exp_names = {etc.name for etc in s.expected_tool_calls}
        for etc in s.expected_tool_calls:
            if etc.name not in tool_names:
                errors.append(f"{s.id}: unknown tool in expected_tool_calls: {etc.name!r}")
        for a, b in s.precedence:
            for name in (a, b):
                if name not in tool_names:
                    errors.append(f"{s.id}: unknown tool in precedence: {name!r}")
                if name not in exp_names:
                    errors.append(f"{s.id}: precedence tool {name!r} not in expected_tool_calls")
        for rule in s.forbidden:
            if rule.name not in tool_names:
                errors.append(f"{s.id}: unknown tool in forbidden: {rule.name!r}")

    if errors:
        bullet = "\n  • "
        raise ValueError(f"Corpus validation failed:{bullet}{bullet.join(errors)}")


def select_scenarios(scenarios: list[Scenario], n: int) -> list[Scenario]:
    """Return n scenarios, stratified across domains via round-robin interleave.

    Within each domain, scenarios are sorted alphabetically by id.
    Domains are processed alphabetically (delivery, emi_reminder, refund).
    The result is deterministic for the same set of scenarios.
    """
    if n >= len(scenarios):
        return list(scenarios)

    by_domain: dict[str, list[Scenario]] = {}
    for s in scenarios:
        by_domain.setdefault(s.domain, []).append(s)
    for lst in by_domain.values():
        lst.sort(key=lambda s: s.id)

    sorted_domains = sorted(by_domain.keys())
    iterators = {d: iter(by_domain[d]) for d in sorted_domains}
    exhausted: set[str] = set()

    result: list[Scenario] = []
    while len(result) < n:
        progress = False
        for domain in sorted_domains:
            if domain in exhausted or len(result) >= n:
                continue
            try:
                result.append(next(iterators[domain]))
                progress = True
            except StopIteration:
                exhausted.add(domain)
        if not progress:
            break

    return result
