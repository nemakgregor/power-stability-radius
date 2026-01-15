import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Allow running the example without installing the package:
# `python examples/script_demo.py` or `poetry run python examples/script_demo.py`
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if _SRC_DIR.is_dir():
    sys.path.insert(0, str(_SRC_DIR))

from stability_radius.dc.dc_model import build_dc_matrices
from stability_radius.parsers.matpower import load_network
from stability_radius.radii.common import get_line_base_quantities
from stability_radius.radii.l2 import compute_l2_radius
from stability_radius.radii.metric import compute_metric_radius
from stability_radius.radii.nminus1 import compute_nminus1_l2_radius
from stability_radius.radii.probabilistic import compute_sigma_radius
from stability_radius.statistics.table import print_radius_summary, print_results_table
from stability_radius.utils import setup_logging
from stability_radius.utils.download import download_ieee_case

logger = logging.getLogger(__name__)

_CONFIG_DIR = (_PROJECT_ROOT / "config").resolve()


def _infer_case_number_from_path(path: str, default: int = 30) -> int:
    """
    Infer IEEE/MATPOWER case number from a filename like 'ieee30.m' or 'case118.m'.

    Parameters
    ----------
    path:
        Input file path.
    default:
        Returned when inference fails.

    Returns
    -------
    int
        Parsed case number or `default`.
    """
    match = re.search(r"(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else default


def _run_tests_or_exit(project_root: Path) -> None:
    """
    Run repository tests before executing the demo.

    This matches the current demo behavior. To skip (e.g., for speed), set:
        SR_SKIP_TESTS=1
    """
    if os.environ.get("SR_SKIP_TESTS", "").strip() == "1":
        return

    cmd = [sys.executable, "-m", "pytest", "-q"]
    completed = subprocess.run(cmd, cwd=str(project_root), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _resolve_under_project_root(p: str) -> str:
    """Resolve a potentially-relative path under the repository root."""
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str((_PROJECT_ROOT / path).resolve())


def _cfg_select(cfg: DictConfig, key: str, default: Any) -> Any:
    """
    Safe config accessor using OmegaConf.select().

    Returns `default` when the key does not exist or cfg is not an OmegaConf object.
    """
    try:
        val = OmegaConf.select(cfg, key)
    except Exception:
        val = None
    return default if val is None else val


def _merge_line_results(
    *dicts: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Merge multiple per-line result dictionaries (same "line_<idx>" keys).

    Later dicts override earlier ones for colliding fields (expected identical base fields).
    """
    keys: set[str] = set()
    for d in dicts:
        keys.update(d.keys())

    merged: dict[str, dict[str, Any]] = {}
    for k in sorted(
        keys, key=lambda s: int(s.split("_", 1)[1]) if "_" in s else 10**18
    ):
        merged[k] = {}
        for d in dicts:
            if k in d:
                merged[k].update(d[k])
    return merged


@hydra.main(config_path=str(_CONFIG_DIR), config_name="main", version_base=None)
def main(cfg: DictConfig) -> None:
    # Configure logs and create runs/<timestamp> directory
    run_dir = setup_logging(cfg)

    # 1) Download case file if it doesn't exist
    case_number = _infer_case_number_from_path(cfg.paths.input, default=30)
    target_path = _resolve_under_project_root(str(cfg.paths.input))
    input_path = download_ieee_case(case_number=case_number, target_path=target_path)

    # 2) Load network (with fallback .m parser if needed)
    net = load_network(input_path)

    # 3) Build DC model matrices (PTDF-like sensitivities)
    H_full, _ = build_dc_matrices(net, slack_bus=int(cfg.settings.slack_bus))
    n_bus = int(H_full.shape[1])

    # 4) Run power flow once and reuse base quantities for all radii
    margin_factor = float(_cfg_select(cfg, "settings.margin_factor", 1.0))
    logger.info(
        "Running base AC PF and extracting base quantities (margin_factor=%.6g)...",
        margin_factor,
    )
    base = get_line_base_quantities(net, margin_factor=margin_factor)

    # 5) Compute all radii

    # 5.1 L2 radii
    l2 = compute_l2_radius(net, H_full, margin_factor=margin_factor, base=base)

    # 5.2 Metric radii: default M = I unless provided by config (not required)
    # If you later add config like settings.metric.diagonal_weight, you can plug it in here.
    M = np.eye(n_bus, dtype=float)
    metric = compute_metric_radius(
        net, H_full, M, margin_factor=margin_factor, base=base
    )

    # 5.3 Probabilistic radii: diagonal Sigma with identical per-bus stddev (default=1 MW)
    inj_std_mw = float(_cfg_select(cfg, "settings.sigma.injection_std_mw", 1.0))
    Sigma_diag = (inj_std_mw**2) * np.ones(n_bus, dtype=float)
    logger.info(
        "Using Sigma as diagonal with injection_std_mw=%.6g (Sigma=std^2*I).",
        inj_std_mw,
    )
    sigma = compute_sigma_radius(
        net, H_full, Sigma_diag, margin_factor=margin_factor, base=base
    )

    # 5.4 Effective N-1 radii via LODF
    update_sens = bool(_cfg_select(cfg, "settings.nminus1.update_sensitivities", True))
    islanding = str(_cfg_select(cfg, "settings.nminus1.islanding", "skip"))
    nminus1 = compute_nminus1_l2_radius(
        net,
        H_full,
        margin_factor=margin_factor,
        update_sensitivities=update_sens,
        islanding=islanding,  # "skip" or "raise"
        base=base,
    )

    # Merge into one per-line dict for JSON/table output
    results = _merge_line_results(l2, metric, sigma, nminus1)

    # 6) Print results table + summaries
    print_results_table(
        results,
        columns=(
            "p0_mw",
            "p_limit_mw_est",
            "margin_mw",
            "norm_g",
            "metric_denom",
            "sigma_flow",
            "radius_l2",
            "radius_metric",
            "radius_sigma",
            "overload_probability",
            "radius_nminus1",
            "worst_contingency",
        ),
    )
    print_radius_summary(results, radius_field="radius_l2")
    print_radius_summary(results, radius_field="radius_metric")
    print_radius_summary(results, radius_field="radius_sigma")
    print_radius_summary(results, radius_field="radius_nminus1")

    # 7) Save results
    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    logger.info("Done. Results written to: %s", results_path)


if __name__ == "__main__":
    _run_tests_or_exit(_PROJECT_ROOT)
    main()
