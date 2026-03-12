from __future__ import annotations

from stability_radius.config import LoggingConfig
from pathlib import Path

def test_setup_logging_creates_run_dir(tmp_path, monkeypatch):
    from stability_radius.utils import setup_logging

    monkeypatch.chdir(tmp_path)

    cfg = LoggingConfig(
        runs_dir="run_artifacts",
        module_name="compute",
        level_console="INFO",
        level_file="DEBUG",
    )
    run_dir = setup_logging(cfg)

    assert (tmp_path / "run_artifacts").exists()
    assert (tmp_path / "run_artifacts").is_dir()
    assert (tmp_path / "run_artifacts" / "compute").joinpath(Path(run_dir).name).exists()
    assert (Path(run_dir) / "debug.log").exists()


def test_create_module_output_dir_normalizes_external_output_under_runs(tmp_path, monkeypatch):
    from stability_radius.utils import create_module_output_dir

    monkeypatch.chdir(tmp_path)

    out_dir = create_module_output_dir(
        module_name="metrics_analysis",
        requested_output_dir="analysis_output/case118_api",
    )

    assert out_dir == (
        tmp_path / "run_artifacts" / "metrics_analysis" / "case118_api"
    ).resolve()


def test_create_module_output_dir_preserves_explicit_runs_path(tmp_path, monkeypatch):
    from stability_radius.utils import create_module_output_dir

    monkeypatch.chdir(tmp_path)

    out_dir = create_module_output_dir(
        module_name="metrics_analysis",
        requested_output_dir="run_artifacts/custom_bucket/report_outputs",
    )

    assert out_dir == (
        tmp_path / "run_artifacts" / "custom_bucket" / "report_outputs"
    ).resolve()
