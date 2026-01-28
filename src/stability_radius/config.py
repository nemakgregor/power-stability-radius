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
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class LoggingConfig:
    """Logging-related defaults for CLI/scripts."""

    runs_dir: str = "runs"
    level_console: str = "INFO"
    level_file: str = "DEBUG"


@dataclass(frozen=True)
class HiGHSConfig:
    """
    HiGHS solver configuration for deterministic OPF.

    Notes
    -----
    - threads=1 avoids non-deterministic parallel execution in some solver builds.
    - random_seed is set explicitly for completeness.
    """

    solver_name: str = "highs"
    threads: int = 1
    random_seed: int = 0

    def solver_options(self) -> dict[str, int]:
        """Return solver options in a PyPSA/linopy-friendly format."""
        return {"threads": int(self.threads), "random_seed": int(self.random_seed)}


@dataclass(frozen=True)
class OPFConfig:
    """Global OPF configuration (base point is always OPF in this project)."""

    highs: HiGHSConfig = field(default_factory=HiGHSConfig)

    # PyPSA requires finite capacities. This value is used as a deterministic surrogate
    # when a line is explicitly unconstrained (+inf or NaN limit).
    unconstrained_line_nom_mw: float = 1e9


@dataclass(frozen=True)
class DCConfig:
    """DC model defaults."""

    mode: str = "operator"  # "operator" or "materialize"
    chunk_size: int = 256
    dtype: str = "float64"  # "float64" or "float32"


@dataclass(frozen=True)
class MonteCarloConfig:
    """Defaults for verification Monte Carlo coverage evaluation."""

    n_samples: int = 50_000
    seed: int = 0
    chunk_size: int = 256

    # Quantile of finite radius_l2 used to scale the sampling box.
    box_radius_quantile: float = 0.10

    # Strict feasibility by default.
    box_feas_tol_mw: float = 0.0

    # Certificate sanity-check (inside min_r ball).
    cert_tol_mw: float = 1e-6
    cert_max_samples: int = 5_000


DEFAULT_LOGGING = LoggingConfig()
DEFAULT_OPF = OPFConfig()
DEFAULT_DC = DCConfig()
DEFAULT_MC = MonteCarloConfig()

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
