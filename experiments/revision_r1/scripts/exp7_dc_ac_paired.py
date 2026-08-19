"""Experiment R1-7: regenerate the paired DC/AC all-constraint radii.

The historical table was computed with the slack-index defect and without the
Q-limit active-set correction, so the AC radii of every case whose ext_grid is
not the first sorted bus (case118_ieee, case200_activ) must be regenerated.
Uses the production workflow (compute_results_for_case) with the sweep's
configuration: base_dispatch=ac_fpf, lossless AC, balanced certificates.

Reports, per case: global DC radius, global AC radius (two-block balanced
[P;Q] certificate, the code's primary output), and the AC/DC ratio.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA_DIR, save_json  # noqa: E402

from stability_radius.base_point.pandapower_tools import (  # noqa: E402
    resolve_slack_bus_id,
)
from stability_radius.parsers.matpower import load_network  # noqa: E402
from stability_radius.utils.download import download_pglib_opf_case  # noqa: E402
from stability_radius.workflows import (  # noqa: E402
    DCExtensionsConfig,
    compute_results_for_case,
)

CASES = [
    "pglib_opf_case5_pjm.m",
    "pglib_opf_case14_ieee.m",
    "pglib_opf_case24_ieee_rts.m",
    "pglib_opf_case30_ieee.m",
    "pglib_opf_case57_ieee.m",
    "pglib_opf_case73_ieee_rts.m",
    "pglib_opf_case118_ieee.m",
    "pglib_opf_case200_activ.m",
]


def global_min(results: dict, key: str) -> float | None:
    vals = []
    for k, v in results.items():
        if not isinstance(v, dict) or key not in v:
            continue
        if v.get("is_unconstrained"):
            continue
        r = float(v[key])
        if np.isfinite(r) and r > 0:
            vals.append(r)
    return float(min(vals)) if vals else None


def run_case(case_file: str) -> dict:
    path = DATA_DIR / case_file
    if not path.exists():
        download_pglib_opf_case(case_filename=case_file, target_path=str(path))
    slack_id = resolve_slack_bus_id(load_network(str(path)), -1)
    res = compute_results_for_case(
        input_path=str(path),
        slack_bus=int(slack_id),  # explicit ext_grid bus id
        base_dispatch="ac_fpf",
        compute_dc=True,
        dc_mode="materialize",
        dc_chunk_size=64,
        dc_dtype=np.dtype("float64"),
        dc_inj_std_mw=10.0,
        dc_extensions=DCExtensionsConfig(probabilistic_enabled=False),
        compute_ac=True,
        ac_chunk_size=64,
        ac_balance=True,
        ac_pf_init="dc",
        ac_pf_solver="pandapower",
        ac_lossless=True,
        allow_download=False,
    )
    label = case_file.replace("pglib_opf_", "").replace(".m", "")
    r_dc = global_min(res, "radius_l2")
    r_ac = global_min(res, "radius_ac_l2")
    nd = sum(
        int(bool(v.get("ac_nondifferentiable_from"))) + int(bool(v.get("ac_nondifferentiable_to")))
        for v in res.values()
        if isinstance(v, dict)
    )
    out = {
        "case": label,
        "r_dc_global": r_dc,
        "r_ac_global": r_ac,
        "ac_over_dc": (r_ac / r_dc) if (r_dc and r_ac) else None,
        "n_nondiff_ends": int(nd),
    }
    print(
        f"{label:18s} DC={r_dc if r_dc else float('nan'):9.3f}  "
        f"AC={r_ac if r_ac else float('nan'):9.3f}  "
        f"AC/DC={out['ac_over_dc'] if out['ac_over_dc'] else float('nan'):6.3f}  ND={nd}"
    )
    return out


def main() -> None:
    rows = []
    for cf in CASES:
        try:
            rows.append(run_case(cf))
        except Exception as e:  # noqa: BLE001
            print(f"{cf}: FAILED {type(e).__name__}: {str(e)[:200]}")
            rows.append({"case": cf, "failed": str(e)[:300]})
    save_json(
        "exp7_dc_ac_paired.json",
        {
            "experiment": "dc_ac_paired_radii",
            "convention": "base_dispatch=ac_fpf, lossless AC, balanced [P;Q] AC certificate, balanced DC PTDF",
            "cases": rows,
        },
    )


if __name__ == "__main__":
    main()
