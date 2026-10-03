"""Generate bar charts from a summary.json produced by `hcb report`."""

from __future__ import annotations

import json
from pathlib import Path


def generate_plots(summary_path: Path, output_dir: Path) -> list[Path]:
    """Read summary.json and write PNG charts to output_dir. Returns paths written.

    Handles partial persona/domain coverage — only plots cells that exist in the data.
    Raises ImportError with install hint if matplotlib is not available.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend; safe in CI and headless envs
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required: pip install 'hinglish-caller-bench[plot]'"
        ) from exc

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    groups: dict[str, dict] = data.get("groups", {})

    # Collect all personas and domains that actually appear in the data.
    personas = sorted(groups.keys())
    domains: set[str] = set()
    for persona_data in groups.values():
        domains.update(persona_data.keys())
    sorted_domains = sorted(domains)

    if not personas or not sorted_domains:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _bar_chart(
        metric: str,
        ci_key: str,
        title: str,
        filename: str,
    ) -> Path | None:
        """Draw and save one grouped-bar chart. Returns Path or None if skipped."""
        n_domains = len(sorted_domains)
        n_personas = len(personas)
        bar_width = 0.8 / n_personas
        x = range(n_domains)

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, persona in enumerate(personas):
            heights = []
            yerr_lo = []
            yerr_hi = []
            for domain in sorted_domains:
                cell = groups.get(persona, {}).get(domain)
                if cell is not None:
                    rate = cell.get(metric, 0.0) or 0.0
                    lo, hi = cell.get(ci_key, [rate, rate])
                    heights.append(rate)
                    yerr_lo.append(max(0.0, rate - lo))
                    yerr_hi.append(max(0.0, hi - rate))
                else:
                    heights.append(0.0)
                    yerr_lo.append(0.0)
                    yerr_hi.append(0.0)

            offsets = [xi + (i - n_personas / 2 + 0.5) * bar_width for xi in x]
            ax.bar(
                offsets,
                heights,
                width=bar_width,
                yerr=[yerr_lo, yerr_hi],
                capsize=4,
                label=persona,
            )

        ax.set_xticks(list(x))
        ax.set_xticklabels(sorted_domains)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_xlabel("Domain")
        ax.set_title(title)
        ax.legend(title="Persona")
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path)
        plt.close(fig)
        return path

    def _pass_at_k_chart() -> Path | None:
        """Draw pass@k chart; returns None if every cell has pass_at_k=None."""
        all_none = all(
            groups.get(p, {}).get(d, {}).get("pass_at_k") is None
            for p in personas
            for d in sorted_domains
        )
        if all_none:
            return None

        n_domains = len(sorted_domains)
        n_personas = len(personas)
        bar_width = 0.8 / n_personas
        x = range(n_domains)

        # Determine k label from first non-None cell
        k_label = "k"
        for p in personas:
            for d in sorted_domains:
                cell = groups.get(p, {}).get(d)
                if cell and cell.get("pass_at_k") is not None:
                    k_label = str(cell.get("k", "k"))
                    break

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, persona in enumerate(personas):
            heights = []
            for domain in sorted_domains:
                cell = groups.get(persona, {}).get(domain)
                val = cell.get("pass_at_k") if cell else None
                heights.append(val if val is not None else 0.0)

            offsets = [xi + (i - n_personas / 2 + 0.5) * bar_width for xi in x]
            ax.bar(offsets, heights, width=bar_width, label=persona)

        ax.set_xticks(list(x))
        ax.set_xticklabels(sorted_domains)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"pass@{k_label}")
        ax.set_xlabel("Domain")
        ax.set_title(f"pass@{k_label} by Persona × Domain")
        ax.legend(title="Persona")
        fig.tight_layout()
        path = output_dir / "pass_at_k.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    p = _bar_chart(
        "success_rate", "success_ci", "Task Success Rate by Persona × Domain", "success_rate.png"
    )
    if p:
        written.append(p)
    p = _bar_chart(
        "tool_rate", "tool_ci", "Tool Correctness Rate by Persona × Domain", "tool_rate.png"
    )
    if p:
        written.append(p)
    p = _pass_at_k_chart()
    if p:
        written.append(p)

    return written
