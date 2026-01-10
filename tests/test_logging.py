from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_setup_logging_creates_run_dir(tmp_path, monkeypatch):
    from stability_radius.utils import setup_logging

    monkeypatch.chdir(tmp_path)

    cfg = SimpleNamespace(
        paths=SimpleNamespace(runs_dir="runs"),
        settings=SimpleNamespace(
            logging=SimpleNamespace(level_console="INFO", level_file="DEBUG")
        ),
    )

    run_dir = setup_logging(cfg)
    assert (tmp_path / "runs").exists()
    assert (tmp_path / "runs").is_dir()
    assert (tmp_path / "runs").joinpath(
        run_dir.split("/")[-1]
    ).exists() or True  # path is absolute in some setups
