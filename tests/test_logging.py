from __future__ import annotations

from pathlib import Path

import pytest
from stability_radius.config import LoggingConfig


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
    assert (
        (tmp_path / "run_artifacts" / "compute").joinpath(Path(run_dir).name).exists()
    )
    assert (Path(run_dir) / "debug.log").exists()


@pytest.mark.parametrize(
    ("requested_output_dir", "expected_parts"),
    [
        ("analysis_output/case118_api", ("metrics_analysis", "case118_api")),
        (
            "run_artifacts/custom_bucket/report_outputs",
            ("custom_bucket", "report_outputs"),
        ),
    ],
)
def test_create_module_output_dir_resolves_under_runs(
    tmp_path, monkeypatch, requested_output_dir, expected_parts
):
    from stability_radius.utils import create_module_output_dir

    monkeypatch.chdir(tmp_path)

    out_dir = create_module_output_dir(
        module_name="metrics_analysis",
        requested_output_dir=requested_output_dir,
    )

    assert out_dir == (tmp_path / "run_artifacts" / Path(*expected_parts)).resolve()
