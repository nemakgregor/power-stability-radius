"""Regenerate revision figures from saved experimental CSV artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

from revision_nonlinear_replay import _plot as plot_nonlinear
from revision_sigma_calibration import _plot as plot_sigma


def _read_csv(path: Path) -> list[dict[str, object]]:
    """Read one saved artifact table into dictionaries."""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    """Render the nonlinear-distance and variance-calibration plots."""
    nonlinear_dir = Path("run_artifacts/revision1_nonlinear_directional")
    nonlinear_rows = _read_csv(nonlinear_dir / "nonlinear_violation_distances.csv")
    for row in nonlinear_rows:
        row["censored_above_max_scale"] = (
            str(row["censored_above_max_scale"]).lower() == "true"
        )
    plot_nonlinear(nonlinear_dir, nonlinear_rows)

    sigma_dir = Path("run_artifacts/revision1_sigma_calibration")
    sigma_rows = _read_csv(sigma_dir / "sigma_line_calibration.csv")
    plot_sigma(sigma_dir, sigma_rows)


if __name__ == "__main__":
    main()
