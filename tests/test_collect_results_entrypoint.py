from __future__ import annotations

import csv
import json
from pathlib import Path

from entry_points.collect_results import _extract_radius_stats, collect


def test_extract_radius_stats_ignores_nonfinite_values() -> None:
    stats = _extract_radius_stats(
        {
            "__meta__": {"input_path": "case.m"},
            "line_0": {
                "radius_l2": 5.0,
                "radius_ac_l2": 4.0,
                "radius_ac_sigma": 3.0,
            },
            "line_1": {
                "radius_l2": float("nan"),
                "radius_ac_l2": float("inf"),
                "radius_ac_sigma": None,
            },
            "line_2": {"radius_l2": 2.0},
        }
    )

    assert stats["n_lines"] == 3
    assert stats["dc"]["count"] == 2
    assert stats["dc"]["min"] == "2"
    assert stats["dc"]["max"] == "5"
    assert stats["ac"]["count"] == 1
    assert stats["sigma"]["count"] == 1


def test_collect_writes_csv_for_case_results(tmp_path: Path) -> None:
    results_dir = tmp_path / "run_artifacts"
    case_dir = results_dir / "run_pglib_sweep"
    case_dir.mkdir(parents=True)

    case_path = case_dir / "pglib_opf_case30_ieee.json"
    case_path.write_text(
        json.dumps(
            {
                "__meta__": {
                    "input_path": "data/input/pglib_opf_case30_ieee.m",
                    "compute_time_sec": 12.5,
                },
                "line_0": {
                    "radius_l2": 2.5,
                    "radius_ac_l2": 1.5,
                    "radius_ac_sigma": 0.8,
                },
                "line_1": {"radius_l2": 5.0},
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "summary.json").write_text("{}", encoding="utf-8")

    csv_path = results_dir / "collect_results" / "all_results.csv"
    collect(results_dir, csv_path)

    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 1
    row = rows[0]
    assert row["experiment"] == "run_pglib_sweep"
    assert row["case"] == "pglib_opf_case30_ieee"
    assert row["input_path"] == "data/input/pglib_opf_case30_ieee.m"
    assert row["n_lines"] == "2"
    assert row["dc_r_min"] == "2.5"
    assert row["ac_r_count"] == "1"
    assert row["sigma_r_count"] == "1"
