"""Experiment 2: AC sigma-radius with UC.jl-derived per-bus injection σ.

Reads ``experiments/configs/uc_jl_case118.yaml``, downloads the UC.jl instance
to extract realistic per-bus σ_P / σ_Q, then computes DC + AC radii
(including sigma-radius) and writes results to
``experiments/output/sigma_radius/``.

Usage::

    python -m experiments.run_sigma_radius
    python -m experiments.run_sigma_radius --config experiments/configs/uc_jl_case118.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from stability_radius.parsers.uc_jl import load_sigma
from stability_radius.utils.download import download_uc_jl_instance
from stability_radius.workflows import (
    ACExtensionsConfig,
    DCExtensionsConfig,
    compute_results_for_case,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "uc_jl_case118.yaml"


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _numpy_serialiser(obj: object) -> object:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def run(config_path: Path) -> None:
    cfg = _load_config(config_path)
    case_cfg = cfg["case"]
    uc_cfg = cfg["uc_jl"]
    compute_cfg = cfg.get("compute", {})
    data_dir = Path(cfg.get("data_dir", "data/input"))
    output_dir = Path(cfg.get("output_dir", "experiments/output/sigma_radius"))
    allow_download = bool(cfg.get("allow_download", False))

    dc_cfg = compute_cfg.get("dc", {})
    ac_cfg = compute_cfg.get("ac", {})
    base_dispatch = str(compute_cfg.get("base_dispatch", "case"))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download UC.jl instance and extract per-bus sigma.
    uc_dest = Path(uc_cfg.get("dest_dir", "data/uc_jl"))
    uc_case_name = str(uc_cfg["case_name"])
    uc_date = str(uc_cfg.get("date", "2017-01-01"))
    power_factor = float(uc_cfg.get("power_factor", 0.9))

    logger.info("Downloading UC.jl instance: %s (date=%s)", uc_case_name, uc_date)
    uc_path = download_uc_jl_instance(
        uc_case_name,
        dest_dir=uc_dest,
        date=uc_date,
    )

    sigma_data = load_sigma(uc_path, power_factor=power_factor)
    sigma_p_mw = sigma_data["sigma_p_mw"]
    sigma_q_mvar = sigma_data["sigma_q_mvar"]

    logger.info(
        "UC.jl sigma loaded: n_bus=%d, σ_P range=[%.4g, %.4g] MW, σ_Q range=[%.4g, %.4g] MVAr",
        len(sigma_p_mw),
        float(np.min(sigma_p_mw)),
        float(np.max(sigma_p_mw)),
        float(np.min(sigma_q_mvar)),
        float(np.max(sigma_q_mvar)),
    )

    # Save sigma arrays for reproducibility.
    sigma_out = output_dir / "sigma_arrays.json"
    with sigma_out.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "sigma_p_mw": sigma_p_mw.tolist(),
                "sigma_q_mvar": sigma_q_mvar.tolist(),
                "metadata": sigma_data["metadata"],
                "n_timesteps": sigma_data["n_timesteps"],
            },
            fh,
            indent=2,
        )
    logger.info("Sigma arrays saved: %s", sigma_out)

    # Step 2: Compute DC + AC radii with uniform sigma (from UC.jl mean).
    # For the AC sigma-radius we use uniform sigma as a baseline,
    # since the workflow currently supports "uniform" source only.
    mean_sigma_p = float(np.mean(sigma_p_mw[sigma_p_mw > 0])) if np.any(sigma_p_mw > 0) else 1.0
    mean_sigma_q = float(np.mean(sigma_q_mvar[sigma_q_mvar > 0])) if np.any(sigma_q_mvar > 0) else 1.0

    input_path = str(data_dir / case_cfg["matpower_file"])
    slack_bus = int(case_cfg.get("slack_bus", 0))

    logger.info(
        "Computing radii for %s (mean σ_P=%.4g MW, mean σ_Q=%.4g MVAr)",
        case_cfg["name"],
        mean_sigma_p,
        mean_sigma_q,
    )

    ac_ext = ACExtensionsConfig(
        sigma_p_mw_source="uniform",
        sigma_q_mvar_source="uniform",
        sigma_p_mw_uniform=mean_sigma_p,
        sigma_q_mvar_uniform=mean_sigma_q,
        metric_enabled=True,
        save_h_vectors=True,
    )

    results = compute_results_for_case(
        input_path=input_path,
        slack_bus=slack_bus,
        base_dispatch=base_dispatch,
        # DC
        compute_dc=True,
        dc_mode=str(dc_cfg.get("mode", "materialize")),
        dc_chunk_size=int(dc_cfg.get("chunk_size", 64)),
        dc_dtype=np.dtype(dc_cfg.get("dtype", "float64")),
        dc_inj_std_mw=float(dc_cfg.get("inj_std_mw", 10.0)),
        dc_extensions=DCExtensionsConfig(probabilistic_enabled=True),
        # AC
        compute_ac=True,
        ac_chunk_size=int(ac_cfg.get("chunk_size", 64)),
        ac_balance=bool(ac_cfg.get("balance", True)),
        ac_pf_init=str(ac_cfg.get("pf_init", "flat")),
        ac_pf_solver=str(ac_cfg.get("pf_solver", "pandapower")),
        ac_lossless=bool(ac_cfg.get("lossless", True)),
        ac_extensions=ac_ext,
        # shared
        allow_download=allow_download,
    )

    # Save h-vectors separately as .npz.
    h_vecs = results.pop("_h_vectors", None)
    if h_vecs is not None:
        npz_path = output_dir / "h_vectors.npz"
        np.savez_compressed(str(npz_path), **h_vecs)
        logger.info("h-vectors saved: %s", npz_path)

    results_path = output_dir / f"{case_cfg['name']}_results.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=_numpy_serialiser)
    logger.info("Results written: %s", results_path)

    # Step 3: Summarise sigma-radius statistics.
    sigma_radii = []
    for key, val in results.items():
        if not key.startswith("line_") or not isinstance(val, dict):
            continue
        r_sig = val.get("radius_ac_sigma")
        if r_sig is not None and np.isfinite(r_sig):
            sigma_radii.append(float(r_sig))

    if sigma_radii:
        logger.info(
            "Sigma-radius stats: n=%d, min=%.4g, median=%.4g, max=%.4g",
            len(sigma_radii),
            min(sigma_radii),
            float(np.median(sigma_radii)),
            max(sigma_radii),
        )

    summary = {
        "case": case_cfg["name"],
        "uc_jl_source": str(uc_path),
        "mean_sigma_p_mw": mean_sigma_p,
        "mean_sigma_q_mvar": mean_sigma_q,
        "n_sigma_radii_finite": len(sigma_radii),
        "sigma_radius_min": min(sigma_radii) if sigma_radii else float("nan"),
        "sigma_radius_median": float(np.median(sigma_radii)) if sigma_radii else float("nan"),
        "sigma_radius_max": max(sigma_radii) if sigma_radii else float("nan"),
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=_numpy_serialiser)
    logger.info("Summary written: %s", summary_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Experiment 2: sigma-radius with UC.jl data.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to uc_jl_case118.yaml config.",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
