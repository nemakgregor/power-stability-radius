"""Experiment 3: Worst-case perturbation verification via nonlinear AC PF.

For each line in a given case, constructs the worst-case perturbation from
the AC L2 certificate, then verifies it with a full pandapower AC power flow.

Requires a prior run of ``run_pglib_sweep.py`` (or equivalent) to produce
a results JSON with h-vectors.

Usage::

    python -m experiments.run_worst_case_verify --results experiments/output/pglib_sweep/pglib_opf_case30_ieee.json
    python -m experiments.run_worst_case_verify --config experiments/configs/pglib_sweep.yaml --case pglib_opf_case30_ieee
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from stability_radius.parsers.matpower import load_network
from stability_radius.verification.verify_worst_case import verify_worst_case
from stability_radius.workflows import _expand_h_reduced_to_full

logger = logging.getLogger(__name__)


def _numpy_serialiser(obj: object) -> object:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def run(
    *,
    results_path: Path,
    h_vectors_path: Path | None = None,
    output_dir: Path | None = None,
    scales: list[float] | None = None,
    top_k: int = 10,
) -> None:
    """Run worst-case verification for lines with smallest AC L2 radii.

    Parameters
    ----------
    results_path:
        Path to a results JSON from ``run_pglib_sweep`` or ``compute_results_for_case``.
    h_vectors_path:
        Optional path to an ``h_vectors.npz`` file.  If None, looks for
        ``h_vectors.npz`` in the same directory as ``results_path``.
    output_dir:
        Output directory.  Defaults to ``experiments/output/worst_case_verify/``.
    scales:
        List of perturbation scale factors to test (e.g. [0.5, 1.0, 1.5]).
    top_k:
        Number of lines (sorted by ascending AC L2 radius) to verify.
    """
    if scales is None:
        scales = [1.0]
    if output_dir is None:
        output_dir = Path("experiments/output/worst_case_verify")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results JSON.
    with results_path.open(encoding="utf-8") as fh:
        results = json.load(fh)

    meta = results.get("__meta__", {})
    input_path = meta.get("input_path")
    slack_bus = int(meta.get("slack_bus", 0))

    if input_path is None:
        raise ValueError("results JSON missing __meta__.input_path")

    # Load h-vectors.
    if h_vectors_path is None:
        h_vectors_path = results_path.parent / "h_vectors.npz"
    if not h_vectors_path.exists():
        raise FileNotFoundError(
            f"h-vectors file not found: {h_vectors_path}.  "
            "Re-run the experiment with save_h_vectors=true."
        )

    hvecs = np.load(str(h_vectors_path))
    h_from_full = hvecs["h_from"]
    h_to_full = hvecs["h_to"]
    bus_ids = hvecs["bus_ids"].tolist()
    line_ids = hvecs["line_ids"].tolist()
    n_bus = len(bus_ids)

    # Load network.
    net = load_network(input_path)

    # Collect per-line AC radii and select top_k smallest.
    line_data: list[dict] = []
    for pos, lid in enumerate(line_ids):
        key = f"line_{lid}"
        row = results.get(key, {})
        r_ac = row.get("radius_ac_l2", float("inf"))
        binding_end = str(row.get("binding_end", "from"))
        s0 = float(row.get(f"ac_s0_{binding_end}_mva", float("nan")))
        limit = float(row.get("ac_s_limit_mva", float("nan")))

        h_vec = h_from_full[pos] if binding_end == "from" else h_to_full[pos]

        if not np.isfinite(r_ac) or r_ac <= 0:
            continue

        line_data.append({
            "pos": pos,
            "line_id": int(lid),
            "radius_ac_l2": float(r_ac),
            "binding_end": binding_end,
            "s0_mva": s0,
            "limit_mva": limit,
            "h_vec": h_vec,
        })

    line_data.sort(key=lambda x: x["radius_ac_l2"])
    selected = line_data[:top_k]

    logger.info(
        "Verifying %d lines (top_k=%d) from %s",
        len(selected),
        top_k,
        results_path.name,
    )

    # Run verification.
    all_results: list[dict] = []

    for entry in selected:
        for scale in scales:
            logger.info(
                "  line=%d radius=%.4g scale=%.2f",
                entry["line_id"],
                entry["radius_ac_l2"],
                scale,
            )

            vr = verify_worst_case(
                net=net,
                line_id=entry["line_id"],
                h_vec=entry["h_vec"],
                radius=entry["radius_ac_l2"],
                s0_mva=entry["s0_mva"],
                limit_mva=entry["limit_mva"],
                scale=scale,
                balance=True,
                lossless=True,
            )

            row = vr.to_dict()
            row["scale"] = scale
            all_results.append(row)

    # Write results.
    case_stem = results_path.stem
    out_path = output_dir / f"{case_stem}_worst_case.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, default=_numpy_serialiser)
    logger.info("Worst-case results written: %s", out_path)

    # Log summary.
    n_verified = sum(1 for r in all_results if r.get("pf_converged"))
    n_violated = sum(1 for r in all_results if r.get("violated"))
    logger.info(
        "Summary: %d verified, %d violated, %d PF failures",
        n_verified,
        n_violated,
        len(all_results) - n_verified,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Experiment 3: worst-case perturbation verification.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to results JSON from run_pglib_sweep or run_sigma_radius.",
    )
    parser.add_argument(
        "--h-vectors",
        type=Path,
        default=None,
        help="Path to h_vectors.npz (default: look in same dir as --results).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/output/worst_case_verify"),
        help="Output directory.",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5],
        help="Perturbation scale factors to test.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of lines (smallest AC L2 radius) to verify.",
    )
    args = parser.parse_args()

    run(
        results_path=args.results,
        h_vectors_path=args.h_vectors,
        output_dir=args.output_dir,
        scales=args.scales,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
