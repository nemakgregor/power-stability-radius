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
    from entry_points.power_stability_radius import build_parser, run_report

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


def test_compute_accepts_ac_nonlinear_validation_flags() -> None:
    from entry_points.power_stability_radius import build_parser

    parser = build_parser(cfg=None)
    args = parser.parse_args(
        [
            "--run-tests",
            "0",
            "compute",
            "--input",
            "data/input/pglib_opf_case30_ieee.m",
            "--ac-validate-nonlinear",
            "1",
            "--ac-validation-top-k",
            "3",
            "--ac-validation-scale-max",
            "2.5",
            "--ac-validation-tol",
            "0.02",
            "--ac-validation-max-iter",
            "7",
        ]
    )

    assert args.ac_validate_nonlinear == 1
    assert args.ac_validation_top_k == 3
    assert args.ac_validation_scale_max == pytest.approx(2.5)
    assert args.ac_validation_tol == pytest.approx(0.02)
    assert args.ac_validation_max_iter == 7
