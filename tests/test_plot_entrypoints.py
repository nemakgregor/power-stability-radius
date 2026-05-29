from __future__ import annotations

import json
from pathlib import Path

from stability_radius.postprocess.plot_radius_distribution import (
    main as radius_distribution_main,
    plot as plot_radius_distribution,
)
from stability_radius.postprocess.plot_sigma_vs_time import plot as plot_sigma_vs_time
from stability_radius.postprocess.plot_worst_case_heatmap import (
    main as worst_case_heatmap_main,
    plot as plot_worst_case_heatmap,
)


def test_plot_radius_distribution_writes_png_and_pdf(tmp_path: Path) -> None:
    input_dir = tmp_path / "run_pglib_sweep"
    input_dir.mkdir()

    for case_name, dc_radius, ac_radius in (
        ("case30", 2.0, 1.0),
        ("case57", 3.0, 1.5),
    ):
        (input_dir / f"pglib_opf_{case_name}.json").write_text(
            json.dumps(
                {
                    "__meta__": {"input_path": f"data/input/{case_name}.m"},
                    "line_0": {"radius_l2": dc_radius, "radius_ac_l2": ac_radius},
                    "line_1": {
                        "radius_l2": dc_radius * 2.0,
                        "radius_ac_l2": ac_radius * 2.0,
                    },
                }
            ),
            encoding="utf-8",
        )

    output_dir = tmp_path / "plots"
    plot_radius_distribution(input_dir, output_dir)

    assert (output_dir / "radius_distribution.pdf").exists()
    assert (output_dir / "radius_distribution.png").exists()


def test_plot_radius_distribution_main_uses_shared_cli(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "pglib_opf_case5_pjm.json").write_text(
        json.dumps(
            {
                "__meta__": {"input_path": "case5.m"},
                "line_0": {"radius_l2": 2.0, "radius_ac_l2": 1.0},
            }
        ),
        encoding="utf-8",
    )

    rc = radius_distribution_main(
        ["--input-dir", str(input_dir), "--output-dir", "plots"]
    )

    output_dir = tmp_path / "run_artifacts" / "plot_radius_distribution" / "plots"
    assert rc == 0
    assert (output_dir / "radius_distribution.pdf").exists()
    assert (output_dir / "debug.log").exists()


def test_plot_sigma_vs_time_writes_expected_outputs(tmp_path: Path) -> None:
    sigma_dir = tmp_path / "run_sigma_radius"
    sigma_dir.mkdir()
    (sigma_dir / "case30_results.json").write_text(
        json.dumps(
            {
                "line_0": {"radius_ac_sigma": 1.0},
                "line_1": {"radius_ac_sigma": 2.5},
            }
        ),
        encoding="utf-8",
    )

    scalability_path = tmp_path / "scalability.json"
    scalability_path.write_text(
        json.dumps(
            [
                {
                    "case": "pglib_opf_case30_ieee",
                    "n_bus": 30,
                    "dc_time_sec_mean": 1.2,
                    "ac_time_sec_mean": 4.5,
                }
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "plots"
    plot_sigma_vs_time(sigma_dir, scalability_path, output_dir)

    assert (output_dir / "sigma_radius_sorted.pdf").exists()
    assert (output_dir / "sigma_vs_time.pdf").exists()


def test_plot_worst_case_heatmap_writes_expected_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "run_worst_case_verify"
    input_dir.mkdir()
    (input_dir / "case30_worst_case.json").write_text(
        json.dumps(
            [
                {
                    "line_id": 0,
                    "scale": 0.9,
                    "relative_error": 0.05,
                    "violated": False,
                    "pf_converged": True,
                    "predicted_s_mva": 95.0,
                    "actual_s_mva": 93.0,
                    "limit_mva": 100.0,
                },
                {
                    "line_id": 0,
                    "scale": 1.0,
                    "relative_error": 0.08,
                    "violated": True,
                    "pf_converged": True,
                    "predicted_s_mva": 101.0,
                    "actual_s_mva": 98.0,
                    "limit_mva": 100.0,
                },
                {
                    "line_id": 1,
                    "scale": 0.9,
                    "relative_error": 0.03,
                    "violated": False,
                    "pf_converged": True,
                    "predicted_s_mva": 80.0,
                    "actual_s_mva": 79.0,
                    "limit_mva": 100.0,
                },
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "plots"
    plot_worst_case_heatmap(input_dir, output_dir)

    assert (output_dir / "worst_case_heatmap.pdf").exists()
    assert (output_dir / "worst_case_scatter.pdf").exists()


def test_plot_worst_case_heatmap_main_uses_shared_cli(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "case5_worst_case.json").write_text(
        json.dumps(
            [
                {
                    "line_id": 0,
                    "scale": 1.0,
                    "relative_error": 0.01,
                    "violated": False,
                    "pf_converged": True,
                    "predicted_s_mva": 90.0,
                    "actual_s_mva": 89.0,
                    "limit_mva": 100.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    rc = worst_case_heatmap_main(
        ["--input-dir", str(input_dir), "--output-dir", "plots"]
    )

    output_dir = tmp_path / "run_artifacts" / "plot_worst_case_heatmap" / "plots"
    assert rc == 0
    assert (output_dir / "worst_case_heatmap.pdf").exists()
    assert (output_dir / "debug.log").exists()
