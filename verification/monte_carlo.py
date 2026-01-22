"""
Monte Carlo coverage estimate for the "true feasibility region" vs. the guaranteed L2 ball.

Workflow (chunked, deterministic)
---------------------------------
- Load results.json
- Extract:
    min_r = min(radius_l2), max_r = max(radius_l2)  (finite only)
    f0    = flow0_mw per line
    c     = p_limit_mw_est per line
- Rebuild DC model from the input MATPOWER/PGLib case
- Sample delta_p uniformly in an axis-aligned box around 0

Important fix (why feasibility was always ~0)
--------------------------------------------
`radius_l2` is an **L2 radius** in injection space. The previous implementation used
per-coordinate bounds `[-2*max_r, 2*max_r]`, which makes the *typical* L2 norm of a
sample scale as `O(max_r * sqrt(n_bus))`. In medium/large dimensions this explodes,
flows become enormous, and the rejection sampler almost never finds feasible points
(total_feasible_in_box becomes 0).

To keep sampling magnitudes consistent with L2 radii we now use:

    box_half_width = 2 * max_r / sqrt(n_bus)

Then the *corner* of the box has ||delta||_2 = 2*max_r, and typical samples have
||delta||_2 = O(max_r), not O(max_r*sqrt(n_bus)).

Feasibility / coverage definitions (unchanged)
----------------------------------------------
- Balance enforced via orthogonalization:
    delta_orth = delta - mean(delta)   (per sample)
- New flows:
    f_new = f0 + DC(delta_orth)
- Feasible:
    feasible = all(abs(f_new) <= c)
- Coverage (conditional on feasibility in the sampling box):
    coverage = (feasible_in_ball / total_feasible_in_box) * 100
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Tuple

import numpy as np

try:
    from scipy.linalg import norm as _scipy_norm  # type: ignore

    _HAVE_SCIPY_NORM = True
except Exception:  # pragma: no cover
    _scipy_norm = None  # type: ignore[assignment]
    _HAVE_SCIPY_NORM = False

# Fix src-layout imports when running as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from stability_radius.dc.dc_model import build_dc_operator
from stability_radius.parsers.matpower import load_network
from stability_radius.radii.common import line_key
from stability_radius.utils import setup_logging

logger = logging.getLogger("verification.monte_carlo")

_Z_95 = 1.959963984540054  # ~N(0,1) quantile for 95% CI


def _configure_logging(level: str, *, runs_dir: str = "runs") -> str:
    """Configure project logging and create a run directory."""
    cfg = SimpleNamespace(
        paths=SimpleNamespace(runs_dir=runs_dir),
        settings=SimpleNamespace(
            logging=SimpleNamespace(level_console=level, level_file="DEBUG")
        ),
    )
    return setup_logging(cfg)


def _load_results(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(obj)}")
    return obj


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
    """
    Extract (f0, c, radius_l2) arrays aligned with sorted net.line.index order.

    Returns
    -------
    (f0, c, r_l2, min_r, max_r)
    """
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

    min_r, max_r = _finite_min_max(r.tolist())
    return f0, c, r, min_r, max_r


def _orthogonalize_balance(delta: np.ndarray) -> np.ndarray:
    """
    Enforce sum(delta)=0 by subtracting the mean across buses (per sample).

    This matches the original task description:
        delta_orth = delta - mean(delta)

    Parameters
    ----------
    delta:
        (batch, n_bus)

    Returns
    -------
    np.ndarray
        Balanced delta (new array).
    """
    if delta.ndim != 2:
        raise ValueError(f"delta must be 2D (batch,n_bus), got {delta.shape}")
    mu = delta.mean(axis=1, keepdims=True)
    return delta - mu


def _norm_rows_l2(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 norm with SciPy if available, else NumPy."""
    if _HAVE_SCIPY_NORM:
        return _scipy_norm(x, axis=1)  # type: ignore[misc]
    return np.linalg.norm(x, ord=2, axis=1)


def _coverage_ci95_percent(*, k: int, n: int) -> tuple[float, float]:
    """
    Simple 95% Wald CI for p = k/n.

    Notes
    -----
    - This is a *diagnostic* for report verification only.
    - For very small n it can be inaccurate; we still clamp to [0, 1] for stability.
    """
    if n <= 0:
        return float("nan"), float("nan")
    p = float(k) / float(n)
    se = math.sqrt(max(p * (1.0 - p), 0.0) / float(n))
    lo = max(0.0, p - _Z_95 * se)
    hi = min(1.0, p + _Z_95 * se)
    return 100.0 * lo, 100.0 * hi


def _sampling_box_bounds(*, max_r: float, n_bus: int) -> tuple[float, float, float]:
    """
    Compute sampling box bounds from an L2 radius scale.

    We interpret `max_r` as an L2 scale in R^n. Using `[-2*max_r, 2*max_r]^n` as
    per-coordinate bounds makes ||delta||_2 explode with sqrt(n_bus), killing
    feasibility probability.

    We instead use:
        half_width = 2*max_r/sqrt(n_bus)

    so that ||delta||_2 <= 2*max_r for all delta in the box.

    Returns
    -------
    (lo, hi, half_width)
    """
    if n_bus <= 0:
        raise ValueError("n_bus must be positive.")
    mr = float(max_r)
    if not math.isfinite(mr) or mr < 0:
        raise ValueError(f"max_r must be finite and non-negative; got {max_r!r}")

    half_width = 0.0 if mr == 0.0 else (2.0 * mr / math.sqrt(float(n_bus)))
    lo = -half_width
    hi = half_width
    return float(lo), float(hi), float(half_width)


def estimate_coverage_percent(
    *,
    results_path: Path,
    input_case_path: Path,
    slack_bus: int,
    n_samples: int = 50_000,
    seed: int = 0,
    chunk_size: int = 256,
) -> Dict[str, Any]:
    """
    Estimate coverage% of the true feasible set inside the guaranteed L2 ball.

    Behavior on hard instances
    --------------------------
    For some large/tight cases, the probability of feasibility inside the sampling box
    may be extremely small. In that situation `total_feasible_in_box` can be 0 for the
    requested `n_samples`. This is not a computational failure, so this function returns:

      - coverage_percent = NaN
      - total_feasible_in_box = 0
      - feasible_in_ball = 0
      - status = "no_feasible_samples"

    Additional fields are returned to make verification reports "fully checkable":
      - n_samples, seed, chunk_size
      - n_bus, n_line
      - box_lo, box_hi, box_half_width, box_half_width_formula
      - feasible_rate_in_box_percent
      - coverage_ci95_low_percent / coverage_ci95_high_percent (only if denom>0)

    Returns
    -------
    dict
        See above; intended to be JSON-serializable.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    logger.debug("Loading results: %s", str(results_path))
    results = _load_results(results_path)

    logger.debug("Loading network: %s", str(input_case_path))
    net = load_network(input_case_path)

    logger.debug("Building DC operator (slack_bus=%s)...", slack_bus)
    dc_op = build_dc_operator(net, slack_bus=int(slack_bus))
    m_line, n_bus = int(dc_op.n_line), int(dc_op.n_bus)
    logger.debug("DC operator dims: n_line=%d, n_bus=%d", m_line, n_bus)

    f0, c, _, min_r, max_r = _extract_radii_f0_c(results=results, net=net)
    if f0.shape != (m_line,) or c.shape != (m_line,):
        raise ValueError(
            f"Shape mismatch: f0={f0.shape}, c={c.shape}, expected ({m_line},)"
        )

    lo, hi, half_width = _sampling_box_bounds(max_r=float(max_r), n_bus=int(n_bus))
    logger.info(
        "Monte Carlo sampling box: half_width=%.6g computed as 2*max_r/sqrt(n_bus) "
        "[max_r=%.6g, n_bus=%d, box=[%.6g, %.6g]]",
        half_width,
        max_r,
        n_bus,
        lo,
        hi,
    )

    rng = np.random.default_rng(seed=int(seed))

    total_feasible = 0
    feasible_in_ball = 0

    remaining = int(n_samples)
    while remaining > 0:
        k = min(int(chunk_size), remaining)
        remaining -= k

        delta = rng.uniform(lo, hi, size=(k, n_bus)).astype(float, copy=False)
        delta_orth = _orthogonalize_balance(delta)

        flow_delta = dc_op.flows_from_delta_injections(delta_orth)  # (k, m)
        f_new = f0[None, :] + flow_delta

        feasible_mask = np.all(np.abs(f_new) <= c[None, :], axis=1)
        num_feasible = int(np.sum(feasible_mask))
        total_feasible += num_feasible

        if num_feasible:
            norms = _norm_rows_l2(delta_orth[feasible_mask, :])
            feasible_in_ball += int(np.sum(norms <= float(min_r)))

    feasible_rate = 100.0 * float(total_feasible) / float(n_samples)

    base_out: Dict[str, Any] = {
        "status": "ok",
        "coverage_percent": float("nan"),
        "coverage_ci95_low_percent": float("nan"),
        "coverage_ci95_high_percent": float("nan"),
        "min_r": float(min_r),
        "max_r": float(max_r),
        "box_lo": float(lo),
        "box_hi": float(hi),
        "box_half_width": float(half_width),
        "box_half_width_formula": "2*max_r/sqrt(n_bus)",
        "n_samples": int(n_samples),
        "seed": int(seed),
        "chunk_size": int(chunk_size),
        "n_bus": int(n_bus),
        "n_line": int(m_line),
        "total_feasible_in_box": int(total_feasible),
        "feasible_in_ball": int(feasible_in_ball),
        "feasible_rate_in_box_percent": float(feasible_rate),
    }

    if total_feasible <= 0:
        # Expected, non-exceptional outcome on tight/large instances.
        logger.info(
            "Monte Carlo coverage: n/a (no feasible samples) "
            "[n_samples=%d, feasible=0, box=[%.6g, %.6g], min_r=%.6g, max_r=%.6g].",
            n_samples,
            lo,
            hi,
            min_r,
            max_r,
        )
        base_out["status"] = "no_feasible_samples"
        return base_out

    coverage = 100.0 * float(feasible_in_ball) / float(total_feasible)
    ci_lo, ci_hi = _coverage_ci95_percent(
        k=int(feasible_in_ball), n=int(total_feasible)
    )

    logger.info(
        "Monte Carlo coverage: %.6f%% (feasible_in_ball=%d / total_feasible_in_box=%d), "
        "feasible_rate_in_box=%.6f%%, min_r=%.6g, max_r=%.6g",
        coverage,
        feasible_in_ball,
        total_feasible,
        feasible_rate,
        min_r,
        max_r,
    )

    base_out["coverage_percent"] = float(coverage)
    base_out["coverage_ci95_low_percent"] = float(ci_lo)
    base_out["coverage_ci95_high_percent"] = float(ci_hi)
    return base_out


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Monte Carlo coverage estimate for feasibility region coverage by the guaranteed L2 ball."
    )
    parser.add_argument(
        "--results", required=True, type=str, help="Path to results.json"
    )
    parser.add_argument(
        "--input", required=True, type=str, help="Path to MATPOWER/PGLib .m case file"
    )
    parser.add_argument(
        "--slack-bus",
        default=0,
        type=int,
        help="Slack bus id or position (consistent with demo)",
    )
    parser.add_argument(
        "--n-samples", default=50_000, type=int, help="Number of Monte Carlo samples"
    )
    parser.add_argument("--seed", default=0, type=int, help="RNG seed")
    parser.add_argument(
        "--chunk-size",
        default=256,
        type=int,
        help="Batch size for vectorized evaluation",
    )
    parser.add_argument("--log-level", default="INFO", type=str, help="Logging level")
    parser.add_argument(
        "--runs-dir",
        default="runs",
        type=str,
        help="Directory where per-run folders and run.log are created.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    _configure_logging(str(args.log_level), runs_dir=str(args.runs_dir))

    stats = estimate_coverage_percent(
        results_path=Path(args.results),
        input_case_path=Path(args.input),
        slack_bus=int(args.slack_bus),
        n_samples=int(args.n_samples),
        seed=int(args.seed),
        chunk_size=int(args.chunk_size),
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
