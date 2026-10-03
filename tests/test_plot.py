"""Tests for src/hinglish_bench/plot.py."""

from __future__ import annotations

import json
from pathlib import Path

from hinglish_bench.plot import generate_plots

FIXTURE = Path(__file__).parent / "fixtures" / "summary.json"


def test_generate_plots_writes_expected_files(tmp_path: Path) -> None:
    paths = generate_plots(FIXTURE, tmp_path)
    names = {p.name for p in paths}
    assert "success_rate.png" in names
    assert "tool_rate.png" in names


def test_pass_at_k_chart_written_when_values_present(tmp_path: Path) -> None:
    paths = generate_plots(FIXTURE, tmp_path)
    names = {p.name for p in paths}
    # Fixture has at least one non-None pass_at_k value (english/refund = 0.333)
    assert "pass_at_k.png" in names


def test_pass_at_k_chart_skipped_when_all_none(tmp_path: Path) -> None:
    data = json.loads(FIXTURE.read_text())
    for persona in data["groups"].values():
        for cell in persona.values():
            cell["pass_at_k"] = None
    p = tmp_path / "no_passk_summary.json"
    p.write_text(json.dumps(data))
    paths = generate_plots(p, tmp_path / "out")
    assert not any(x.name == "pass_at_k.png" for x in paths)


def test_plot_tolerates_partial_groups(tmp_path: Path) -> None:
    # Fixture deliberately has only 2 personas × 2 domains — must not raise
    paths = generate_plots(FIXTURE, tmp_path)
    assert len(paths) >= 2


def test_output_dir_is_created(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "plots"
    assert not out.exists()
    generate_plots(FIXTURE, out)
    assert out.exists()


def test_files_are_valid_pngs(tmp_path: Path) -> None:
    paths = generate_plots(FIXTURE, tmp_path)
    for p in paths:
        assert p.suffix == ".png"
        # PNG magic bytes: \x89PNG
        assert p.read_bytes()[:4] == b"\x89PNG"


def test_cli_plot_missing_summary(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from hinglish_bench.cli import app

    result = CliRunner().invoke(app, ["plot", str(tmp_path)])
    assert result.exit_code == 1
    assert "summary.json not found" in result.output


def test_cli_plot_writes_to_default_subdir(tmp_path: Path) -> None:
    import shutil

    from typer.testing import CliRunner

    from hinglish_bench.cli import app

    # Copy fixture into a fake results dir
    results = tmp_path / "results"
    results.mkdir()
    shutil.copy(FIXTURE, results / "summary.json")

    result = CliRunner().invoke(app, ["plot", str(results)])
    assert result.exit_code == 0, result.output
    assert (results / "plots" / "success_rate.png").exists()
