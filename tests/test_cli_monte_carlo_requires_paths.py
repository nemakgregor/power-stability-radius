from __future__ import annotations

from pathlib import Path

import pytest


def test_report_requires_cases_in_config_and_fails_fast_without_side_effects(
    tmp_path: Path, monkeypatch
) -> None:
    """
    CLI regression:
    - `report` must not silently assume a hard-coded case list.
    - If cfg_loaded is None (no YAML), it must fail before creating run artifacts.
    """
    from stability_radius.cli import build_parser, run_report

    monkeypatch.chdir(tmp_path)

    parser = build_parser(cfg=None)
    args = parser.parse_args(
        [
            "--runs-dir",
            "runs",
            "--run-tests",
            "0",
            "report",
            "--results-dir",
            "verification/results",
            "--out",
            "verification/report.md",
        ]
    )

    with pytest.raises(ValueError, match=r"report requires a loaded YAML config"):
        run_report(
            args, cfg_loaded=None, cfg_path=tmp_path / "cfg.yaml", argv=["report"]
        )

    assert not (tmp_path / "runs").exists()
