from __future__ import annotations

from pathlib import Path

import pytest


def _set_home(monkeypatch: pytest.MonkeyPatch, home_dir: Path) -> None:
    home_dir.mkdir(parents=True, exist_ok=True)

    # POSIX
    monkeypatch.setenv("HOME", str(home_dir))

    # Windows (best-effort, harmless on POSIX)
    monkeypatch.setenv("USERPROFILE", str(home_dir))


def test_setup_logging_expands_tilde_runs_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stability_radius.config import LoggingConfig
    from stability_radius.utils import setup_logging

    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    monkeypatch.chdir(tmp_path)

    cfg = LoggingConfig(
        runs_dir="~/run_artifacts",
        level_console="INFO",
        level_file="DEBUG",
        run_dir_mode="overwrite",
        run_name="run1",
    )
    run_dir = Path(setup_logging(cfg)).resolve()

    expected = (home / "run_artifacts" / "general" / "run1").resolve()
    assert run_dir == expected
    assert expected.is_dir()
    assert (expected / "debug.log").exists()


def test_cli_resolve_path_expands_tilde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from entry_points.power_stability_radius import _resolve_path

    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    monkeypatch.chdir(tmp_path)

    resolved = Path(_resolve_path("~/x/results.json")).resolve()
    assert resolved == (home / "x" / "results.json").resolve()


def test_ensure_case_file_expands_tilde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stability_radius.utils.download import ensure_case_file

    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    monkeypatch.chdir(tmp_path)

    case_path = home / "case14.m"
    case_path.write_text("existing", encoding="utf-8")

    out = Path(ensure_case_file("~/case14.m")).resolve()
    assert out == case_path.resolve()
