from __future__ import annotations

from pathlib import Path


def test_ci_workflow_runs_formatting_and_tests() -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "poetry install" in workflow
    assert "ruff format --check ." in workflow
    assert "python -m pytest" in workflow
    assert "--cov=src/stability_radius" in workflow
    assert "--cov=entry_points" not in workflow
