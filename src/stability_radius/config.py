from __future__ import annotations

"""
Central configuration for the project.

Why this module exists
----------------------
The repository previously had many duplicated "DEFAULT_*" constants scattered across CLI,
verification scripts, and library code. This makes behavior drift likely and complicates
auditing determinism.

This module centralizes all user-facing defaults and solver settings, so:
- CLI defaults == verification defaults
- OPF solver is enforced globally (PyPSA + HiGHS)
- Monte Carlo defaults are consistent across entrypoints

YAML config loading
-------------------
The project uses Hydra-style (OmegaConf-compatible) YAML files under `conf/`.
To keep dependencies lightweight and avoid bringing the full Hydra runtime,
we support a minimal composition mechanism:

- `extends: <path-or-list>` at the top level of a YAML file.
- `extends` paths are resolved relative to the extending file.
- Configs are merged deterministically in the given order, where later configs override earlier ones.

This enables "experiment" configs that only override a few keys while inheriting the main defaults.

Numerical stability note (HiGHS / scaling)
------------------------------------------
This project enforces an OPF->DCOperator *consistency check*:
we reconstruct line flows from OPF bus injections using the same DCOperator model and
expect the flows to match PyPSA-reported base flows within a tight tolerance.

If the LP is poorly scaled (e.g., huge surrogate line limits like 1e9 and huge penalty
costs like 1e6), HiGHS can return a solution with small primal feasibility drift that
is enough to fail that consistency check at the MW level.

Defaults below are chosen to avoid common scaling pitfalls and to enable HiGHS internal
scaling when it warns about "excessively large costs / row bounds".
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Tuple

logger = logging.getLogger(__name__)

try:
    from omegaconf import OmegaConf  # type: ignore

    HAVE_OMEGACONF: bool = True
except Exception:  # noqa: BLE001
    OmegaConf = None  # type: ignore[assignment]
    HAVE_OMEGACONF = False


@dataclass(frozen=True)
class LoggingConfig:
    """
    Logging- and run-directory-related defaults for CLI/scripts.

    Notes
    -----
    `setup_logging(...)` creates a run directory and configures:
    - a project file logger (runs/<run>/run.log)
    - a console logger

    Run directory mode
    ------------------
    - run_dir_mode="timestamp": create a new unique folder per run (default).
    - run_dir_mode="overwrite": reuse `runs_dir/run_name` (delete/recreate folder).
    """

    runs_dir: str = "runs"
    level_console: str = "INFO"
    level_file: str = "DEBUG"

    # Output folder management.
    run_dir_mode: str = "timestamp"  # "timestamp" | "overwrite"
    run_name: str = "latest"  # used only when run_dir_mode="overwrite"


@dataclass(frozen=True)
class HiGHSConfig:
    """
    HiGHS solver configuration for deterministic OPF.

    Notes
    -----
    - threads=1 avoids non-deterministic parallel execution in some solver builds.
    - random_seed is set explicitly for completeness.

    Scaling / numerical stability
    -----------------------------
    HiGHS can warn about "excessively large costs / row bounds" on poorly scaled models.
    Such cases are common in power-grid LPs if we represent "+inf" constraints via huge
    surrogates (e.g., s_nom=1e9) and/or use huge penalty costs.

    Those warnings are not cosmetic: small primal infeasibilities can later manifest as
    OPF->DCOperator flow mismatches at the MW level.

    The default options below follow HiGHS recommendations from its own warning messages:
    - user_objective_scale=-1: let HiGHS auto-scale objective
    - user_bound_scale=-10: scale bounds/rows (helps with huge RHS values)
    - tighter feasibility tolerances for better self-consistency of angles/flows
    """

    solver_name: str = "highs"
    threads: int = 1
    random_seed: int = 42

    # Recommended by HiGHS when it detects poor scaling:
    user_objective_scale: int = -1
    user_bound_scale: int = -10

    # Optional, but improves self-consistency of the returned primal solution.
    primal_feasibility_tolerance: float = 1e-9
    dual_feasibility_tolerance: float = 1e-9

    def solver_options(self) -> dict[str, Any]:
        """
        Return solver options in a PyPSA/linopy-friendly format.

        Returns
        -------
        dict[str, Any]
            Options passed verbatim into HiGHS via linopy. Names match HiGHS option names.
        """
        return {
            "threads": int(self.threads),
            "random_seed": int(self.random_seed),
            "user_objective_scale": int(self.user_objective_scale),
            "user_bound_scale": int(self.user_bound_scale),
            "primal_feasibility_tolerance": float(self.primal_feasibility_tolerance),
            "dual_feasibility_tolerance": float(self.dual_feasibility_tolerance),
        }


@dataclass(frozen=True)
class OPFConfig:
    """Global OPF configuration (base point is always OPF in this project)."""

    highs: HiGHSConfig = field(default_factory=HiGHSConfig)

    # PyPSA requires finite capacities. This value is used as a deterministic surrogate
    # when a line is explicitly unconstrained (+inf or NaN limit).
    #
    # IMPORTANT: Keep it reasonably large, but avoid astronomically large values that
    # destroy LP scaling (HiGHS will warn and the OPF->DCOperator consistency check can fail).
    unconstrained_line_nom_mw: float = 1e6


@dataclass(frozen=True)
class DCConfig:
    """DC model defaults."""

    mode: str = "operator"  # "operator" or "materialize"
    chunk_size: int = 256
    dtype: str = "float64"  # "float64" or "float32"


@dataclass(frozen=True)
class MonteCarloConfig:
    """
    Defaults for verification Monte Carlo evaluation.

    What is evaluated
    -----------------
    1) Soundness (certificate check):
       uniform sampling in the certified L2 ball of radius r* in the balanced subspace,
       verifying that all line constraints hold.

    2) Probabilistic safety (Gaussian injections):
       balanced injections with i.i.d. per-bus sigma (taken from results.json meta by default),
       estimating P(feasible) via MC and computing the analytic lower bound P(||Δp||<=r*).

    Legacy note
    -----------
    box_radius_quantile is a legacy parameter from the old "box vs ball coverage" experiment.
    The current verification workflow does NOT use box sampling for reporting/validation.
    """

    n_samples: int = 50_000
    seed: int = 0
    chunk_size: int = 256

    # Legacy (deprecated): kept for CLI backward-compatibility; ignored by current MC.
    box_radius_quantile: float = 0.10

    # Strict feasibility by default (used as feasibility tolerance in MW).
    box_feas_tol_mw: float = 0.0

    # Certificate sanity-check (inside min_r ball).
    cert_tol_mw: float = 1e-6
    cert_max_samples: int = 5_000


@dataclass(frozen=True)
class UnitsConfig:
    """
    Unit/physics validation policy.

    strict_units
    ------------
    When True, the library fails fast on non-physical or ambiguous quantities instead of
    silently producing 0.0 coefficients.

    Enforced conditions (high level)
    --------------------------------
    - vn_kv > 0
    - x_ohm > 0 (for lossless DC branches)
    - sn_mva > 0 (for transformer and system base quantities)

    allow_phase_shift
    -----------------
    When False, transformers with non-zero shift_degree are rejected.

    Rationale:
    - The project's DCOperator/PTDF model currently ignores phase shifters.
    - Ignoring shift_degree without an explicit opt-in leads to silent OPF/DC inconsistencies.
    """

    strict_units: bool = True
    allow_phase_shift: bool = False


DEFAULT_LOGGING = LoggingConfig()
DEFAULT_OPF = OPFConfig()
DEFAULT_DC = DCConfig()
DEFAULT_MC = MonteCarloConfig()
DEFAULT_UNITS = UnitsConfig()

DEFAULT_TABLE_COLUMNS: Tuple[str, ...] = (
    "flow0_mw",
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
    "worst_contingency_line_idx",
)

DEFAULT_NMINUS1_ISLANDING: str = "skip"


def _resolve_path(p: str | Path, *, base_dir: Path | None) -> Path:
    """Resolve a potentially-relative path against `base_dir` (or CWD if base_dir is None)."""
    path = Path(p).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = base_dir if base_dir is not None else Path.cwd()
    return (root / path).resolve()


def _as_list(value: Any) -> list[str]:
    """Normalize a scalar/list config node into a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for x in value:
            if x is None:
                continue
            sx = str(x).strip()
            if sx:
                out.append(sx)
        return out
    raise TypeError(f"extends must be a string or a list of strings; got {type(value)}")


def _load_with_extends(path: Path, *, stack: tuple[Path, ...]) -> Any:
    """
    Internal recursive loader for `extends` composition with cycle detection.

    Parameters
    ----------
    path:
        Absolute path to a YAML config file.
    stack:
        Current recursion stack used for deterministic cycle diagnostics.

    Returns
    -------
    Any
        OmegaConf config object (DictConfig).
    """
    if not HAVE_OMEGACONF or OmegaConf is None:  # pragma: no cover - guarded by caller
        raise ImportError(
            "OmegaConf is required to load YAML configs (install `hydra-core`)."
        )

    p = path.resolve()
    if p in stack:
        chain = " -> ".join([*(str(x) for x in stack), str(p)])
        raise ValueError(f"Cyclic config extends detected: {chain}")

    cfg_local = OmegaConf.load(str(p))

    extends_raw = OmegaConf.select(cfg_local, "extends")
    extends_list = _as_list(extends_raw)

    base_cfgs: list[Any] = []
    for ext in extends_list:
        base_path = _resolve_path(ext, base_dir=p.parent)
        if not base_path.exists():
            raise FileNotFoundError(
                f"Extended config not found: {base_path} (referenced from {p})"
            )
        base_cfgs.append(_load_with_extends(base_path, stack=(*stack, p)))

    # Remove 'extends' from the local config before merge to keep the effective config clean.
    local_container = OmegaConf.to_container(cfg_local, resolve=False)
    if not isinstance(local_container, dict):
        raise ValueError(
            f"Config root must be a mapping/object, got {type(local_container)} in {p}"
        )
    local_container.pop("extends", None)
    cfg_no_ext = OmegaConf.create(local_container)

    merged = OmegaConf.merge(*base_cfgs, cfg_no_ext) if base_cfgs else cfg_no_ext

    logger.debug(
        "Loaded config: %s (extends=%s)",
        str(p),
        extends_list if extends_list else "[]",
    )
    return merged


def load_project_config(path: str | Path, *, allow_missing: bool = True) -> Any:
    """
    Load a project YAML config with minimal inheritance support via `extends`.

    Parameters
    ----------
    path:
        Path to the YAML config file. Can be relative (resolved against current working directory).
    allow_missing:
        If True and the file does not exist, returns None (caller may fall back to Python defaults).

    Returns
    -------
    Any
        OmegaConf config object (DictConfig) or None if allow_missing=True and file is missing.

    Raises
    ------
    ImportError
        If OmegaConf is not installed and the file exists.
    FileNotFoundError
        If allow_missing=False and the file does not exist, or if an `extends` target is missing.
    ValueError
        On cyclic `extends` chains or invalid config shapes.
    """
    cfg_path = _resolve_path(path, base_dir=None)

    if not cfg_path.exists():
        if allow_missing:
            logger.info(
                "Config file not found, using built-in defaults: %s", str(cfg_path)
            )
            return None
        raise FileNotFoundError(str(cfg_path))

    if not HAVE_OMEGACONF or OmegaConf is None:
        raise ImportError(
            "OmegaConf is required to load YAML configs (install `hydra-core`)."
        )

    return _load_with_extends(cfg_path, stack=())
