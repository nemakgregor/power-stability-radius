from __future__ import annotations

from stability_radius.config import LoggingConfig
from pathlib import Path

def test_setup_logging_creates_run_dir(tmp_path, monkeypatch):
    from stability_radius.utils import setup_logging

    monkeypatch.chdir(tmp_path)

    cfg = LoggingConfig(runs_dir="runs", level_console="INFO", level_file="DEBUG")
    run_dir = setup_logging(cfg)

    assert (tmp_path / "runs").exists()
    assert (tmp_path / "runs").is_dir()
    assert (tmp_path / "runs").joinpath(Path(run_dir).name).exists()
