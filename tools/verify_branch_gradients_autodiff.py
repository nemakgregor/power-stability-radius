"""Independent automatic-differentiation check of two-ended AC branch gradients."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Any

import autograd.numpy as anp
import numpy as np
from autograd import jacobian

from stability_radius.base_point.ac import solve_ac_pf_base_point
from stability_radius.parsers.matpower import load_network


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _flow_components(
    x: anp.ndarray, *, self_g: float, self_b: float, cross_g: float, cross_b: float
) -> anp.ndarray:
    """Return P, Q, and |S| for one branch end in per unit."""
    theta_i, theta_k, vm_i, vm_k = x
    delta = theta_i - theta_k
    p = (
        self_g * vm_i**2
        + vm_i * vm_k * (cross_g * anp.cos(delta) + cross_b * anp.sin(delta))
    )
    q = (
        -self_b * vm_i**2
        + vm_i * vm_k * (cross_g * anp.sin(delta) - cross_b * anp.cos(delta))
    )
    return anp.array([p, q, anp.sqrt(p * p + q * q)])


def _auto_slack_bus(net: Any) -> int:
    for idx in sorted(net.ext_grid.index):
        row = net.ext_grid.loc[idx]
        if bool(row.get("in_service", True)):
            return int(row["bus"])
    return int(sorted(net.bus.index)[0])


def _case_rows(case_path: Path) -> list[dict[str, Any]]:
    from pandapower.converter import to_ppc
    from pandapower.pypower.dSbr_dV import dSbr_dV
    from pandapower.pypower.idx_brch import F_BUS, T_BUS
    from pandapower.pypower.idx_bus import VA, VM
    from pandapower.pypower.makeYbus import makeYbus

    net = load_network(str(case_path))
    slack_bus = _auto_slack_bus(net)
    _bp, base_pf = solve_ac_pf_base_point(
        net=net,
        slack_bus=slack_bus,
        pf_solver="pandapower",
        pf_init="dc",
        lossless=False,
        gen_dispatch_mw_by_name={},
        distributed_slack=False,
        trafo_model="pi",
    )

    nn = copy.deepcopy(net)
    ppc = to_ppc(
        nn,
        calculate_voltage_angles=True,
        trafo_model="pi",
        voltage_depend_loads=False,
        init="flat",
    )
    base_mva = float(ppc["baseMVA"])
    bus_ids = [int(x) for x in sorted(nn.bus.index)]
    bus_lookup = np.asarray(nn._pd2ppc_lookups["bus"], dtype=int)
    ppc_bus_pos = np.asarray([int(bus_lookup[bid]) for bid in bus_ids], dtype=int)
    ppc["bus"][ppc_bus_pos, VM] = np.asarray(base_pf.v_mag_pu, dtype=float)
    ppc["bus"][ppc_bus_pos, VA] = np.rad2deg(
        np.asarray(base_pf.v_ang_rad, dtype=float)
    )
    voltage = ppc["bus"][:, VM] * np.exp(1j * np.deg2rad(ppc["bus"][:, VA]))
    _ybus, yf, yt = makeYbus(base_mva, ppc["bus"], ppc["branch"])
    dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm, sf, st = dSbr_dV(
        ppc["branch"], yf, yt, voltage
    )

    start, stop = nn._pd2ppc_lookups["branch"]["line"]
    net_line_order = [int(x) for x in nn.line.index]
    if int(stop) - int(start) != len(net_line_order):
        raise ValueError("Unexpected pandapower line lookup length.")

    rows: list[dict[str, Any]] = []
    for local_pos, line_id in enumerate(net_line_order):
        branch_row = int(start) + int(local_pos)
        f = int(ppc["branch"][branch_row, F_BUS].real)
        t = int(ppc["branch"][branch_row, T_BUS].real)
        for line_end in ("from", "to"):
            if line_end == "from":
                i, k = f, t
                yrow = yf.getrow(branch_row)
                dS_dVa, dS_dVm, flow = dSf_dVa, dSf_dVm, sf
            else:
                i, k = t, f
                yrow = yt.getrow(branch_row)
                dS_dVa, dS_dVm, flow = dSt_dVa, dSt_dVm, st
            coefficients = {int(col): complex(value) for col, value in zip(yrow.indices, yrow.data)}
            y_self = coefficients.get(i, 0.0 + 0.0j)
            y_cross = coefficients.get(k, 0.0 + 0.0j)
            x0 = np.asarray(
                [
                    np.angle(voltage[i]),
                    np.angle(voltage[k]),
                    abs(voltage[i]),
                    abs(voltage[k]),
                ],
                dtype=float,
            )
            ad_jacobian = np.asarray(
                jacobian(
                    lambda x: _flow_components(
                        x,
                        self_g=float(y_self.real),
                        self_b=float(y_self.imag),
                        cross_g=float(y_cross.real),
                        cross_b=float(y_cross.imag),
                    )
                )(x0),
                dtype=float,
            ) * base_mva

            analytic_p = np.asarray(
                [
                    dS_dVa[branch_row, i].real,
                    dS_dVa[branch_row, k].real,
                    dS_dVm[branch_row, i].real,
                    dS_dVm[branch_row, k].real,
                ],
                dtype=float,
            ) * base_mva
            analytic_q = np.asarray(
                [
                    dS_dVa[branch_row, i].imag,
                    dS_dVa[branch_row, k].imag,
                    dS_dVm[branch_row, i].imag,
                    dS_dVm[branch_row, k].imag,
                ],
                dtype=float,
            ) * base_mva
            p0 = float((flow[branch_row] * base_mva).real)
            q0 = float((flow[branch_row] * base_mva).imag)
            s0 = math.hypot(p0, q0)
            if s0 > 1e-12:
                analytic_s = (p0 * analytic_p + q0 * analytic_q) / s0
                s_error = float(np.max(np.abs(ad_jacobian[2] - analytic_s)))
            else:
                s_error = float("nan")
            rows.append(
                {
                    "case": case_path.stem,
                    "line_id": int(line_id),
                    "line_end": line_end,
                    "max_abs_error_p": float(np.max(np.abs(ad_jacobian[0] - analytic_p))),
                    "max_abs_error_q": float(np.max(np.abs(ad_jacobian[1] - analytic_q))),
                    "max_abs_error_s": s_error,
                    "base_flow_mva": s0,
                    "base_mva": base_mva,
                }
            )
    return rows


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    all_rows: list[dict[str, Any]] = []
    for case_name in args.cases:
        all_rows.extend(_case_rows(Path(args.data_dir) / case_name))
    _write_csv(output_dir / "branch_gradient_autodiff.csv", all_rows)
    finite_s = [row["max_abs_error_s"] for row in all_rows if math.isfinite(row["max_abs_error_s"])]
    summary = {
        "cases": list(args.cases),
        "line_ends": len(all_rows),
        "max_abs_error_p": max(row["max_abs_error_p"] for row in all_rows),
        "max_abs_error_q": max(row["max_abs_error_q"] for row in all_rows),
        "max_abs_error_s": max(finite_s) if finite_s else float("nan"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "branch_gradient_autodiff_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/input")
    parser.add_argument("--output-dir", default="run_artifacts/revision1_autodiff")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["pglib_opf_case14_ieee.m", "pglib_opf_case30_ieee.m"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
