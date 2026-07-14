"""Unified CLI for Visual Eval pipeline.

Usage:
    visual-eval t2i generate --models full
    visual-eval t2i judge
    visual-eval t2i aggregate
    visual-eval t2i report
    visual-eval edit run --models sanity --dry-run
    visual-eval edit judge
    visual-eval edit aggregate
    visual-eval dashboard
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(
    name="visual-eval",
    help="Unified evaluation pipeline for T2I generation and image editing models.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
t2i_app = typer.Typer(help="Text-to-image evaluation commands.", no_args_is_help=True)
edit_app = typer.Typer(help="Image editing evaluation commands.", no_args_is_help=True)
app.add_typer(t2i_app, name="t2i")
app.add_typer(edit_app, name="edit")


def _run_module(module: str, extra_args: list[str] | None = None):
    cmd = [sys.executable, "-m", module] + (extra_args or [])
    raise SystemExit(subprocess.call(cmd))


# ─── T2I Commands ──────────────────────────────────────────────


@t2i_app.command()
def generate(
    models: str = typer.Option("full", help="Model profile: sanity, full, or all"),
    layer: int = typer.Option(0, help="Prompt layer (0=all, 1=gold, 2=proprietary)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without running"),
):
    """Generate images across selected T2I models."""
    args = ["--models", models]
    if layer:
        args += ["--layer", str(layer)]
    if dry_run:
        args.append("--dry-run")
    _run_module("scripts.t2i.run_generation", args)


@t2i_app.command("judge")
def t2i_judge(
    models: str = typer.Option("", help="Comma-separated models to judge (empty=all)"),
):
    """Run MLLM judge on generated images."""
    args = ["--models", models] if models else []
    _run_module("scripts.t2i.run_judge", args)


@t2i_app.command("aggregate")
def t2i_aggregate():
    """Aggregate judgment scores into leaderboard CSVs."""
    _run_module("scripts.t2i.run_aggregate")


@t2i_app.command("report")
def t2i_report():
    """Generate PDF reports and per-model scorecards."""
    _run_module("scripts.t2i.run_report")


@t2i_app.command()
def prompts():
    """Build the prompt set (L1+L2+L3)."""
    _run_module("scripts.t2i.run_prompt_set")


@t2i_app.command()
def hitl():
    """Launch human-in-the-loop validation web UI."""
    _run_module("scripts.t2i.run_hitl")


# ─── Edit Commands ─────────────────────────────────────────────


@edit_app.command("run")
def edit_run(
    models: str = typer.Option("full", help="Model profile: sanity, available, or full"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without running"),
):
    """Run image edits across selected editing models."""
    args = ["--models", models]
    if dry_run:
        args.append("--dry-run")
    _run_module("scripts.edit.run_edit", args)


@edit_app.command("judge")
def edit_judge():
    """Run dual-image MLLM judge on edited images."""
    _run_module("scripts.edit.run_judge")


@edit_app.command("aggregate")
def edit_aggregate():
    """Aggregate edit scores into leaderboard CSVs."""
    _run_module("scripts.edit.run_aggregate")


@edit_app.command("report")
def edit_report():
    """Generate edit evaluation report."""
    _run_module("scripts.edit.run_report")


@edit_app.command()
def download_images():
    """Download source images for edit evaluation."""
    _run_module("scripts.edit.download_source_images")


# ─── Top-level Commands ───────────────────────────────────────


@app.command()
def dashboard():
    """Launch the interactive Streamlit results dashboard."""
    dashboard_path = Path(__file__).parent.parent / "dashboard" / "app.py"
    if not dashboard_path.exists():
        typer.echo(f"Dashboard not found at {dashboard_path}", err=True)
        raise typer.Exit(1)
    _run_module("streamlit", ["run", str(dashboard_path)])


@app.command()
def test():
    """Run the test suite."""
    _run_module("pytest", ["tests/", "-v"])


if __name__ == "__main__":
    app()
