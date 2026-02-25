"""
Central configuration for the project.

This module centralizes all user-facing defaults and solver settings, so:
- CLI defaults == verification defaults
- OPF solver is enforced globally (PyPSA + HiGHS)
- Monte Carlo defaults are consistent across entrypoints

YAML config loading
-------------------
The project uses Hydra-style (OmegaConf-compatible) YAML files under `conf/`.
We support a minimal deterministic composition mechanism via `extends`.

Determinism contract (important)
--------------------------------
Some parameters must be identical across entrypoints:
- Programmatic usage: DEFAULT_* dataclasses from this module
- CLI/YAML usage: composed config chain from conf/config.yaml

If these diverge, CI tests and CLI experiments can silently use different limits.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple

from omegaconf import OmegaConf  # type: ignore

logger = logging.getLogger(__name__)

HAVE_OMEGACONF: bool = True

# NOTE:
# This value MUST match `conf/config_shared.yaml: opf.unconstrained_line_nom_mw`.
# It is used as a finite surrogate for "unconstrained" thermal limits in PyPSA.
DEFAULT_UNCONSTRAINED_LINE_NOM_MW: float = 1.0e5


@dataclass(frozen=True)
class LoggingConfig:
    """Logging defaults for CLI/scripts."""

    runs_dir: str = "runs"
    level_console: str = "INFO"
    level_file: str = "DEBUG"
    run_dir_mode: str = "timestamp"  # "timestamp" | "overwrite"
    run_name: str = "latest"  # used only when run_dir_mode="overwrite"


@dataclass(frozen=True)
class HiGHSConfig:
    """HiGHS solver configuration for deterministic OPF."""

    solver_name: str = "highs"
    threads: int = 1
    random_seed: int = 42
    user_objective_scale: int = -1
    user_bound_scale: int = -10
    primal_feasibility_tolerance: float = 1e-9
    dual_feasibility_tolerance: float = 1e-9

    def solver_options(self) -> dict[str, Any]:
        """Return solver options in a linopy/HiGHS-friendly format."""
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
    """
    Global OPF configuration (dispatch source when compute.base_dispatch=dc_opf).

    Notes
    -----
    - The project solves DC OPF (PyPSA + HiGHS) for dispatch only.
    - AC certificate is built around AC PF base point (not AC OPF).

    Reproducibility contract
    ------------------------
    Defaults MUST match the composed YAML defaults (conf/config_shared.yaml),
    because the project supports both:
    - programmatic usage (DEFAULT_OPF), and
    - CLI/YAML usage.
    """

    highs: HiGHSConfig = field(default_factory=HiGHSConfig)

    # MUST match conf/config_shared.yaml (reproducibility across entrypoints).
    unconstrained_line_nom_mw: float = DEFAULT_UNCONSTRAINED_LINE_NOM_MW

    # MUST match conf/config_shared.yaml (security margin policy).
    headroom_factor: float = 0.98

    # moved from module constant into config (user-tunable, deterministic)
    ext_grid_marginal_cost_base: float = 1000.0


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

    Reproducibility contract
    ------------------------
    seed MUST match conf/config_monte_carlo.yaml (and report defaults),
    otherwise CLI (YAML) vs programmatic runs diverge silently.
    """

    n_samples: int = 50_000
    seed: int = 42
    chunk_size: int = 256
    feas_tol_mw: float = 0.0
    cert_tol_mw: float = 1e-6
    cert_max_samples: int = 5_000


DEFAULT_LOGGING = LoggingConfig()
DEFAULT_OPF = OPFConfig()
DEFAULT_DC = DCConfig()
DEFAULT_MC = MonteCarloConfig()

# Flat-table defaults (CLI "table --format flat").
# Note: AC-only results should not use this directly; the CLI now infers AC defaults.
DEFAULT_TABLE_COLUMNS: Tuple[str, ...] = (
    "flow0_mw",
    "p0_mw",
    "p_limit_mw_est",
    "margin_mw",
    "norm_g",
    "radius_l2",
)

DEFAULT_NMINUS1_ISLANDING: str = "skip"


def _resolve_path(p: str | Path, *, base_dir: Path | None) -> Path:
    """Resolve a potentially-relative path against base_dir (or CWD)."""
    path = Path(p).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = base_dir if base_dir is not None else Path.cwd()
    return (root / path).resolve()


def _as_list(value: Any) -> list[str]:
    """
    Normalize a scalar/list config node into a list of strings.

    Notes
    -----
    OmegaConf uses its own container types:
    - ListConfig for YAML sequences
    - DictConfig for mappings

    `extends:` in YAML is typically a sequence -> ListConfig, so we must treat any
    non-string Sequence as a list of paths.
    """
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []

    if isinstance(value, SequenceABC) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        out: list[str] = []
        for x in value:
            if x is None:
                continue
            sx = str(x).strip()
            if sx:
                out.append(sx)
        return out

    raise TypeError(
        "extends must be a string or a list of strings; "
        f"got {type(value)} with value={value!r}"
    )


def _load_with_extends(path: Path, *, stack: tuple[Path, ...]) -> Any:
    """Internal recursive loader for `extends` composition with cycle detection."""
    if not HAVE_OMEGACONF or OmegaConf is None:  # pragma: no cover
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

    local_container = OmegaConf.to_container(cfg_local, resolve=False)
    if not isinstance(local_container, dict):
        raise ValueError(
            f"Config root must be a mapping/object, got {type(local_container)} in {p}"
        )
    local_container.pop("extends", None)
    cfg_no_ext = OmegaConf.create(local_container)

    merged = OmegaConf.merge(*base_cfgs, cfg_no_ext) if base_cfgs else cfg_no_ext
    logger.debug(
        "Loaded config: %s (extends=%s)", str(p), extends_list if extends_list else "[]"
    )
    return merged


def load_project_config(path: str | Path, *, allow_missing: bool = True) -> Any:
    """
    Load a project YAML config with minimal inheritance support via `extends`.
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
