from __future__ import annotations

"""
Monte Carlo verification for the DC L2-ball robustness certificate.

What is verified
----------------
1) Soundness (certificate check):
   sample uniformly inside the certified L2 ball of radius r* (in the balanced subspace)
   and verify that all line flow limits hold.

2) Probabilistic safety (Gaussian injections):
   sample balanced Gaussian injections Δp ~ N(0, σ^2 I) (projected to balanced subspace)
   and estimate P(feasible). Also compute the analytic lower bound:
     P(||Δp|| <= r*) = F_{χ²(d)}((r*/σ)^2), d = n_bus - 1.

Legacy note
-----------
Older versions estimated "coverage" via sampling a cube and measuring the fraction of
feasible samples inside the L2 ball. That quantity is not robust in high dimension and
is no longer used for validation/reporting.

Determinism notes
-----------------
- All randomness is controlled via an explicit `seed`.
- No implicit scaling heuristics beyond explicit parameters (kept only for backward CLI compatibility).

Input data policy
-----------------
- If an expected input `.m` case file is missing and its filename matches a supported
  dataset (MATPOWER case<N>.m/ieee<N>.m or PGLib-OPF pglib_opf_*.m), it is downloaded
  deterministically (stable URL candidate ordering).
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np

# Fix src-layout imports when running as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from stability_radius.config import DEFAULT_LOGGING, DEFAULT_MC, LoggingConfig
from stability_radius.dc.dc_model import build_dc_operator
from stability_radius.parsers.matpower import load_network
from stability_radius.radii.common import line_key
from stability_radius.utils import log_stage, setup_logging
from stability_radius.utils.download import ensure_case_file

logger = logging.getLogger("stability_radius.verification.monte_carlo")

_Z_95 = 1.959963984540054  # ~N(0,1) quantile for 95% CI

_EPS_DIR_NORM = 1e-12


def _configure_logging(level: str, *, runs_dir: str = "runs") -> str:
    """Configure project logging and create a run directory."""
    return setup_logging(
        LoggingConfig(
            runs_dir=str(runs_dir), level_console=str(level), level_file="DEBUG"
        )
    )


def _load_results(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(obj)}")
    return obj


def _get_meta(results: Dict[str, Any]) -> Dict[str, Any]:
    meta = results.get("__meta__")
    return meta if isinstance(meta, dict) else {}


def _finite_min_max(values: Iterable[float]) -> Tuple[float, float]:
    finite = [float(x) for x in values if math.isfinite(float(x))]
    if not finite:
        raise ValueError("No finite values found.")
    return float(min(finite)), float(max(finite))


def _extract_radii_f0_c(
    *,
    results: Dict[str, Any],
    net,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Extract (f0, c, radius_l2) arrays aligned with sorted net.line.index order."""
    line_indices = sorted(net.line.index)

    f0 = np.empty(len(line_indices), dtype=float)
    c = np.empty(len(line_indices), dtype=float)
    r = np.empty(len(line_indices), dtype=float)

    missing: list[str] = []
    for pos, lid in enumerate(line_indices):
        k = line_key(int(lid))
        row = results.get(k)
        if not isinstance(row, dict):
            missing.append(k)
            continue

        try:
            f0[pos] = float(row["flow0_mw"])
            c[pos] = float(row["p_limit_mw_est"])
            r[pos] = float(row["radius_l2"])
        except KeyError as e:
            raise KeyError(f"Missing required field {e} in {k}") from e

    if missing:
        raise KeyError(
            f"results.json does not contain required line keys (first 10): {missing[:10]}"
        )

    if np.isnan(c).any():
        bad = np.where(np.isnan(c))[0]
        raise ValueError(
            "results.json contains NaN in p_limit_mw_est; feasibility becomes undefined. "
            f"Bad line positions count={int(bad.size)} (first 10: {bad[:10].tolist()})."
        )

    min_r, max_r = _finite_min_max(r.tolist())
    return f0, c, r, min_r, max_r


def _orthogonalize_balance(delta: np.ndarray) -> np.ndarray:
    """Enforce sum(delta)=0 by subtracting the mean across buses (per sample)."""
    if delta.ndim != 2:
        raise ValueError(f"delta must be 2D (batch,n_bus), got {delta.shape}")
    mu = delta.mean(axis=1, keepdims=True)
    return delta - mu


def _norm_rows_l2(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 norm."""
    return np.linalg.norm(x, ord=2, axis=1)


def _wald_ci95_percent(*, k: int, n: int) -> tuple[float, float]:
    """Simple 95% Wald CI for p = k/n."""
    if n <= 0:
        return float("nan"), float("nan")
    p = float(k) / float(n)
    se = math.sqrt(max(p * (1.0 - p), 0.0) / float(n))
    lo = max(0.0, p - _Z_95 * se)
    hi = min(1.0, p + _Z_95 * se)
    return 100.0 * lo, 100.0 * hi


def _chi2_cdf(*, x: float, df: int) -> float:
    """
    Chi-square CDF using SciPy special function.

    CDF for χ²_k at x:
        F(x) = gammainc(k/2, x/2)
    """
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}")
    xx = float(x)
    if not math.isfinite(xx) or xx <= 0.0:
        return 0.0

    try:
        from scipy.special import gammainc  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError("SciPy is required to compute chi-square CDF.") from e

    return float(gammainc(float(df) / 2.0, xx / 2.0))


def _sample_balanced_gaussian(
    *,
    rng: np.random.Generator,
    n: int,
    n_bus: int,
    sigma_mw: float,
) -> np.ndarray:
    """
    Sample balanced Gaussian injections.

    Procedure:
    - z ~ N(0, sigma^2 I) in R^n
    - project to balanced subspace by subtracting row-wise mean
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if n_bus <= 1:
        raise ValueError("n_bus must be >= 2 for balanced sampling.")
    s = float(sigma_mw)
    if not math.isfinite(s) or s <= 0.0:
        raise ValueError(f"sigma_mw must be finite and positive; got {sigma_mw!r}")

    z = (s * rng.standard_normal(size=(n, n_bus))).astype(float, copy=False)
    return _orthogonalize_balance(z)


def _sample_balanced_uniform_l2_ball(
    *,
    rng: np.random.Generator,
    n: int,
    n_bus: int,
    radius: float,
) -> np.ndarray:
    """
    Sample uniformly from the L2 ball of radius `radius` in the balanced subspace sum(delta)=0.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if n_bus <= 1:
        raise ValueError("n_bus must be >= 2 for balanced sampling.")
    r = float(radius)
    if not math.isfinite(r) or r < 0:
        raise ValueError(f"radius must be finite and non-negative; got {r!r}")
    if r == 0.0:
        return np.zeros((n, n_bus), dtype=float)

    z = rng.standard_normal(size=(n, n_bus)).astype(float, copy=False)
    z = _orthogonalize_balance(z)
    norms = _norm_rows_l2(z)

    bad = norms <= _EPS_DIR_NORM
    while bool(np.any(bad)):
        k_bad = int(np.sum(bad))
        z_bad = rng.standard_normal(size=(k_bad, n_bus)).astype(float, copy=False)
        z_bad = _orthogonalize_balance(z_bad)
        z[bad, :] = z_bad
        norms = _norm_rows_l2(z)
        bad = norms <= _EPS_DIR_NORM

    dirs = z / norms[:, None]

    d = int(n_bus - 1)
    u = rng.random(size=n).astype(float, copy=False)
    rad = r * np.power(u, 1.0 / float(d))
    return dirs * rad[:, None]


def estimate_coverage_percent(
    *,
    results_path: Path,
    input_case_path: Path,
    slack_bus: int,
    n_samples: int = DEFAULT_MC.n_samples,
    seed: int = DEFAULT_MC.seed,
    chunk_size: int = DEFAULT_MC.chunk_size,
    box_radius_quantile: float = DEFAULT_MC.box_radius_quantile,  # legacy (ignored)
    box_feas_tol_mw: float = DEFAULT_MC.box_feas_tol_mw,
    cert_tol_mw: float = DEFAULT_MC.cert_tol_mw,
    cert_max_samples: int = DEFAULT_MC.cert_max_samples,
) -> Dict[str, Any]:
    """
    Monte Carlo verification entrypoint.

    Output fields (high-level)
    --------------------------
    - base_point_*: feasibility of the OPF base flow w.r.t. stored limits
    - gaussian_*: probabilistic safety metrics for Δp~N(0,σ^2 I) (balanced)
    - certificate_*: soundness check inside the certified L2 ball (radius r*)
    """
    # box_radius_quantile is kept only for backward-compatibility with older CLI/report scripts.
    _ = float(box_radius_quantile)

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    tol_feas = float(box_feas_tol_mw)
    if not math.isfinite(tol_feas) or tol_feas < 0:
        raise ValueError("box_feas_tol_mw must be finite and non-negative.")

    tol_cert = float(cert_tol_mw)
    if not math.isfinite(tol_cert) or tol_cert < 0:
        raise ValueError("cert_tol_mw must be finite and non-negative.")

    cert_max = int(cert_max_samples)
    if cert_max < 0:
        raise ValueError("cert_max_samples must be non-negative.")

    with log_stage(logger, "Read Results (results.json)"):
        results = _load_results(results_path)
        meta = _get_meta(results)

    with log_stage(logger, "Ensure input case file (download if missing)"):
        ensured = ensure_case_file(str(input_case_path))
        input_case_path_eff = Path(ensured)

    with log_stage(logger, "Read Data (MATPOWER/PGLib -> pandapower)"):
        net = load_network(input_case_path_eff)

    line_indices = [int(x) for x in sorted(net.line.index)]

    with log_stage(logger, f"Build DC Model (DCOperator, slack_bus={slack_bus})"):
        dc_op = build_dc_operator(net, slack_bus=int(slack_bus))
        m_line, n_bus = int(dc_op.n_line), int(dc_op.n_bus)
        d_bal = int(max(n_bus - 1, 0))
        logger.debug(
            "DC operator dims: n_line=%d, n_bus=%d, d_balance=%d", m_line, n_bus, d_bal
        )

    f0, c, r, min_r, max_r = _extract_radii_f0_c(results=results, net=net)
    if f0.shape != (m_line,) or c.shape != (m_line,):
        raise ValueError(
            f"Shape mismatch: f0={f0.shape}, c={c.shape}, expected ({m_line},)"
        )

    # Gaussian sigma comes from results meta (generated together with radius_sigma).
    sigma_meta = meta.get("inj_std_mw", None)
    if sigma_meta is None:
        raise ValueError(
            "results.json is missing __meta__.inj_std_mw which is required for Gaussian verification. "
            "Regenerate results with the current pipeline."
        )
    try:
        sigma_mw = float(sigma_meta)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid __meta__.inj_std_mw={sigma_meta!r}") from e
    if not math.isfinite(sigma_mw) or sigma_mw <= 0.0:
        raise ValueError(f"__meta__.inj_std_mw must be finite and >0, got {sigma_mw!r}")

    base_viol = np.where(np.abs(f0) > (c + tol_feas))[0]
    base_feasible = bool(base_viol.size == 0)
    base_max_violation = (
        float(np.max(np.abs(f0[base_viol]) - c[base_viol])) if base_viol.size else 0.0
    )

    if not base_feasible:
        logger.warning(
            "Base point is infeasible w.r.t. stored limits: violated=%d/%d, max_violation=%.6g MW.",
            int(base_viol.size),
            int(f0.size),
            base_max_violation,
        )

    # Analytic lower bound: P(||Δp|| <= r*) under balanced N(0, sigma^2 I)
    if math.isfinite(float(min_r)) and float(min_r) >= 0.0:
        x = (float(min_r) / float(sigma_mw)) ** 2
        ball_mass = 100.0 * _chi2_cdf(x=x, df=max(int(n_bus - 1), 1))
    else:
        ball_mass = float("nan")

    logger.info(
        "Gaussian verification setup: sigma_mw=%.6g, r*=min_r=%.6g, dim=d=n_bus-1=%d, "
        "analytic_P(||dp||<=r*)=%.6g%%",
        float(sigma_mw),
        float(min_r),
        int(max(n_bus - 1, 0)),
        float(ball_mass) if math.isfinite(ball_mass) else float("nan"),
    )

    rng = np.random.default_rng(seed=int(seed))

    # Gaussian MC: P(feasible) and empirical P(||dp||<=r*)
    gauss_feasible = 0
    gauss_in_ball = 0
    gauss_in_ball_and_feasible = 0

    gauss_worst_max_violation = float("-inf")
    gauss_worst_line_pos = -1
    gauss_worst_line_idx = -1
    gauss_worst_sample_l2 = float("nan")

    with log_stage(logger, "Monte Carlo: Gaussian feasibility (balanced)"):
        remaining = int(n_samples)
        while remaining > 0:
            k = min(int(chunk_size), remaining)
            remaining -= k

            delta = _sample_balanced_gaussian(
                rng=rng, n=k, n_bus=int(n_bus), sigma_mw=float(sigma_mw)
            )

            norms = _norm_rows_l2(delta)
            gauss_in_ball += int(np.sum(norms <= float(min_r)))

            viol = dc_op.flows_from_delta_injections(delta)  # (k, m)
            viol += f0[None, :]
            np.abs(viol, out=viol)
            viol -= c[None, :]  # abs(flow) - limit

            np.nan_to_num(
                viol,
                copy=False,
                nan=float("inf"),
                posinf=float("inf"),
                neginf=-float("inf"),
            )

            max_v = np.max(viol, axis=1)
            feasible_mask = max_v <= float(tol_feas)

            gauss_feasible += int(np.sum(feasible_mask))
            if int(np.sum(feasible_mask)) > 0:
                gauss_in_ball_and_feasible += int(
                    np.sum((norms <= float(min_r)) & feasible_mask)
                )

            batch_worst = float(np.max(max_v))
            if batch_worst > gauss_worst_max_violation:
                gauss_worst_max_violation = batch_worst
                j = int(np.argmax(max_v))
                gauss_worst_sample_l2 = float(norms[j])
                lp = int(np.argmax(viol[j, :]))
                gauss_worst_line_pos = lp
                gauss_worst_line_idx = (
                    int(line_indices[lp]) if 0 <= lp < len(line_indices) else -1
                )

    gauss_feasible_percent = 100.0 * float(gauss_feasible) / float(n_samples)
    gauss_ci_lo, gauss_ci_hi = _wald_ci95_percent(
        k=int(gauss_feasible), n=int(n_samples)
    )

    gauss_in_ball_percent = 100.0 * float(gauss_in_ball) / float(n_samples)
    ball_ci_lo, ball_ci_hi = _wald_ci95_percent(k=int(gauss_in_ball), n=int(n_samples))

    # Prepare base output
    out: Dict[str, Any] = {
        "status": "ok",
        "n_samples": int(n_samples),
        "seed": int(seed),
        "chunk_size": int(chunk_size),
        "n_bus": int(n_bus),
        "n_line": int(m_line),
        "dim_balance": int(d_bal),
        "min_r": float(min_r),
        "max_r": float(max_r),
        "base_point_feasible": bool(base_feasible),
        "base_point_violated_lines": int(base_viol.size),
        "base_point_max_violation_mw": float(base_max_violation),
        "feas_tol_mw": float(tol_feas),
        # Gaussian
        "gaussian_sigma_mw": float(sigma_mw),
        "gaussian_feasible_samples": int(gauss_feasible),
        "gaussian_feasible_percent": float(gauss_feasible_percent),
        "gaussian_feasible_ci95_low_percent": float(gauss_ci_lo),
        "gaussian_feasible_ci95_high_percent": float(gauss_ci_hi),
        "gaussian_in_ball_samples": int(gauss_in_ball),
        "gaussian_in_ball_percent": float(gauss_in_ball_percent),
        "gaussian_in_ball_ci95_low_percent": float(ball_ci_lo),
        "gaussian_in_ball_ci95_high_percent": float(ball_ci_hi),
        "gaussian_in_ball_and_feasible_samples": int(gauss_in_ball_and_feasible),
        "gaussian_ball_mass_analytic_percent": float(ball_mass),
        "gaussian_worst_max_violation_mw": float(gauss_worst_max_violation),
        "gaussian_worst_max_violation_line_pos": int(gauss_worst_line_pos),
        "gaussian_worst_max_violation_line_idx": int(gauss_worst_line_idx),
        "gaussian_worst_sample_l2": float(gauss_worst_sample_l2),
        # Certificate check settings
        "certificate_tol_mw": float(tol_cert),
        "certificate_max_samples": int(cert_max),
    }

    # Derive status
    if not base_feasible:
        out["status"] = "base_point_infeasible"

    # Soundness check inside the certified ball
    n_cert = min(int(n_samples), int(cert_max))
    if n_cert > 0:
        with log_stage(logger, "Monte Carlo: L2-ball certificate soundness check (DC)"):
            cert = _run_l2_ball_certificate_check(
                dc_op=dc_op,
                f0=f0,
                c=c,
                min_r=float(min_r),
                n_bus=int(n_bus),
                line_indices=line_indices,
                seed=int(seed) + 1_000_003,
                n_samples=int(n_cert),
                chunk_size=int(chunk_size),
                tol_mw=float(tol_cert),
                base_point_feasible=bool(base_feasible),
            )
        out.update(cert)

        if (
            out.get("certificate_status") == "violations_found"
            and out["status"] == "ok"
        ):
            out["status"] = "certificate_violations_found"

    # Cross-check: if certificate is sound and base point is feasible, all in-ball
    # samples must be feasible. We log this as a diagnostic (do not fail hard here,
    # certificate check above is already the main hard signal).
    if base_feasible and int(gauss_in_ball) > 0:
        if gauss_in_ball_and_feasible != gauss_in_ball:
            logger.warning(
                "Gaussian diagnostic: not all samples inside the certified ball were feasible "
                "(in_ball=%d, in_ball_and_feasible=%d). This suggests a certificate/model inconsistency.",
                int(gauss_in_ball),
                int(gauss_in_ball_and_feasible),
            )
        else:
            logger.debug(
                "Gaussian diagnostic: all in-ball samples were feasible (count=%d).",
                int(gauss_in_ball),
            )

    logger.info(
        "Gaussian MC: P(feasible)=%.6g%% (=%d/%d), "
        "P(||dp||<=r*)_mc=%.6g%% (=%d/%d), P(||dp||<=r*)_analytic=%.6g%%",
        float(gauss_feasible_percent),
        int(gauss_feasible),
        int(n_samples),
        float(gauss_in_ball_percent),
        int(gauss_in_ball),
        int(n_samples),
        float(ball_mass) if math.isfinite(ball_mass) else float("nan"),
    )
    return out


def _run_l2_ball_certificate_check(
    *,
    dc_op,
    f0: np.ndarray,
    c: np.ndarray,
    min_r: float,
    n_bus: int,
    line_indices: list[int],
    seed: int,
    n_samples: int,
    chunk_size: int,
    tol_mw: float,
    base_point_feasible: bool,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "certificate_status": "skipped",
        "certificate_samples": 0,
        "certificate_feasible_samples": 0,
        "certificate_violation_samples": 0,
        "certificate_max_violation_mw": float("nan"),
        "certificate_max_violation_line_pos": -1,
        "certificate_max_violation_line_idx": -1,
        "certificate_tol_mw": float(tol_mw),
        "certificate_dim": int(max(n_bus - 1, 0)),
    }

    if n_samples <= 0:
        return out

    if not base_point_feasible:
        out["certificate_status"] = "base_point_infeasible"
        out["certificate_samples"] = int(n_samples)
        out["certificate_violation_samples"] = int(n_samples)
        return out

    if not math.isfinite(float(min_r)):
        out["certificate_status"] = "invalid_min_r"
        out["certificate_samples"] = int(n_samples)
        out["certificate_violation_samples"] = int(n_samples)
        return out

    if float(min_r) <= 0.0:
        out["certificate_status"] = "trivial_radius_zero"
        out["certificate_samples"] = int(n_samples)
        out["certificate_feasible_samples"] = int(n_samples)
        out["certificate_violation_samples"] = 0
        out["certificate_max_violation_mw"] = float("-inf")
        return out

    rng = np.random.default_rng(int(seed))

    worst_max_violation = float("-inf")
    worst_line_pos = -1
    worst_line_idx = -1

    feasible = 0
    total = 0

    remaining = int(n_samples)
    while remaining > 0:
        k = min(int(chunk_size), remaining)
        remaining -= k

        delta = _sample_balanced_uniform_l2_ball(
            rng=rng, n=k, n_bus=int(n_bus), radius=float(min_r)
        )

        viol = dc_op.flows_from_delta_injections(delta)  # (k, m)
        viol += f0[None, :]
        np.abs(viol, out=viol)
        viol -= c[None, :]  # abs(flow) - limit

        np.nan_to_num(
            viol,
            copy=False,
            nan=float("inf"),
            posinf=float("inf"),
            neginf=-float("inf"),
        )

        max_v = np.max(viol, axis=1)
        feasible_mask = max_v <= float(tol_mw)

        feasible += int(np.sum(feasible_mask))
        total += int(k)

        batch_worst = float(np.max(max_v))
        if batch_worst > worst_max_violation:
            worst_max_violation = batch_worst
            j = int(np.argmax(max_v))
            lp = int(np.argmax(viol[j, :]))
            worst_line_pos = lp
            worst_line_idx = (
                int(line_indices[lp]) if 0 <= lp < len(line_indices) else -1
            )

    out["certificate_samples"] = int(total)
    out["certificate_feasible_samples"] = int(feasible)
    out["certificate_violation_samples"] = int(total - feasible)
    out["certificate_max_violation_mw"] = float(worst_max_violation)
    out["certificate_max_violation_line_pos"] = int(worst_line_pos)
    out["certificate_max_violation_line_idx"] = int(worst_line_idx)
    out["certificate_status"] = "ok" if feasible == total else "violations_found"
    return out


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Monte Carlo verification: Gaussian feasibility + analytic certified ball mass "
            "+ L2-ball certificate soundness check."
        )
    )
    parser.add_argument(
        "--results", required=True, type=str, help="Path to results.json"
    )
    parser.add_argument(
        "--input", required=True, type=str, help="Path to MATPOWER/PGLib .m case file"
    )
    parser.add_argument("--slack-bus", default=0, type=int)
    parser.add_argument("--n-samples", default=DEFAULT_MC.n_samples, type=int)
    parser.add_argument("--seed", default=DEFAULT_MC.seed, type=int)
    parser.add_argument("--chunk-size", default=DEFAULT_MC.chunk_size, type=int)

    # Deprecated args: kept for CLI backward-compatibility
    parser.add_argument(
        "--box-radius-quantile",
        default=DEFAULT_MC.box_radius_quantile,
        type=float,
        help="DEPRECATED (ignored): legacy parameter from old box-based coverage experiment.",
    )

    parser.add_argument(
        "--box-feas-tol-mw",
        default=DEFAULT_MC.box_feas_tol_mw,
        type=float,
        help="Feasibility tolerance in MW (applies to Gaussian feasibility checks).",
    )
    parser.add_argument("--cert-tol-mw", default=DEFAULT_MC.cert_tol_mw, type=float)
    parser.add_argument(
        "--cert-max-samples", default=DEFAULT_MC.cert_max_samples, type=int
    )
    parser.add_argument("--log-level", default=DEFAULT_LOGGING.level_console, type=str)
    parser.add_argument("--runs-dir", default=DEFAULT_LOGGING.runs_dir, type=str)
    args = parser.parse_args(list(argv) if argv is not None else None)

    _configure_logging(str(args.log_level), runs_dir=str(args.runs_dir))

    stats = estimate_coverage_percent(
        results_path=Path(args.results),
        input_case_path=Path(args.input),
        slack_bus=int(args.slack_bus),
        n_samples=int(args.n_samples),
        seed=int(args.seed),
        chunk_size=int(args.chunk_size),
        box_radius_quantile=float(args.box_radius_quantile),  # ignored
        box_feas_tol_mw=float(args.box_feas_tol_mw),
        cert_tol_mw=float(args.cert_tol_mw),
        cert_max_samples=int(args.cert_max_samples),
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
