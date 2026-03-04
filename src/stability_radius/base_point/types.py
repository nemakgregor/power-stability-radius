from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BasePointDC:
    """
    DC base regime.

    Notes
    -----
    This dataclass is meant to be JSON-friendly via `to_meta_dict()`.

    Units
    -----
    - bus_injections_mw: MW
    - line_flows_mw: MW
    - line_limits_mw: MW (usually derived from MVA under PF=1 convention)
    """

    source: str  # "case" | "dc_opf"
    slack_bus: int

    bus_ids: tuple[int, ...]
    bus_injections_mw: np.ndarray  # (n_bus,)

    line_ids: tuple[int, ...]
    line_flows_mw: np.ndarray  # (n_line,)
    line_limits_mw: np.ndarray  # (n_line,)

    status: str
    objective: float

    # Dispatch used to reproduce AC PF regime (PyPSA generator names -> P(MW)).
    gen_dispatch_mw_by_name: tuple[tuple[str, float], ...] = ()

    def to_meta_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict (arrays -> lists)."""
        return {
            "source": str(self.source),
            "slack_bus": int(self.slack_bus),
            "bus_ids": [int(x) for x in self.bus_ids],
            "bus_injections_mw": [
                float(x)
                for x in np.asarray(self.bus_injections_mw, dtype=float).tolist()
            ],
            "line_ids": [int(x) for x in self.line_ids],
            "line_flows_mw": [
                float(x) for x in np.asarray(self.line_flows_mw, dtype=float).tolist()
            ],
            "line_limits_mw": [
                float(x) for x in np.asarray(self.line_limits_mw, dtype=float).tolist()
            ],
            "status": str(self.status),
            "objective": float(self.objective),
            "gen_dispatch_mw_by_name": [
                (str(k), float(v)) for k, v in self.gen_dispatch_mw_by_name
            ],
        }


@dataclass(frozen=True)
class BasePointAC:
    """
    AC base regime around an AC PF solution.

    Units
    -----
    - vm_pu: p.u.
    - va_rad: rad
    - line flows: MW / MVAr
    - line_s_limit_mva: MVA
    """

    pf_solver: str  # "pandapower" | "pypsa"
    pf_init: str  # "flat" | "dc" | "pp"
    lossless: bool
    slack_bus: int

    bus_ids: tuple[int, ...]
    vm_pu: np.ndarray  # (n_bus,)
    va_rad: np.ndarray  # (n_bus,)

    line_ids: tuple[int, ...]
    p_from_mw: np.ndarray  # (n_line,)
    q_from_mvar: np.ndarray  # (n_line,)
    p_to_mw: np.ndarray  # (n_line,)
    q_to_mvar: np.ndarray  # (n_line,)
    s_limit_mva: np.ndarray  # (n_line,)

    status: str

    # For reproducibility: DC OPF dispatch used to form this PF regime (if any).
    gen_dispatch_mw_by_name: tuple[tuple[str, float], ...] = ()

    # AC PF repair metadata: which solver attempt succeeded and what was changed.
    pf_attempt: str = "primary"  # "primary" | "alt_init" | "relaxed"
    pf_repairs: tuple[str, ...] = ()  # list of repair actions applied

    def to_meta_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict (arrays -> lists)."""
        return {
            "pf_solver": str(self.pf_solver),
            "pf_init": str(self.pf_init),
            "lossless": bool(self.lossless),
            "slack_bus": int(self.slack_bus),
            "bus_ids": [int(x) for x in self.bus_ids],
            "vm_pu": [float(x) for x in np.asarray(self.vm_pu, dtype=float).tolist()],
            "va_rad": [float(x) for x in np.asarray(self.va_rad, dtype=float).tolist()],
            "line_ids": [int(x) for x in self.line_ids],
            "s_limit_mva": [
                float(x) for x in np.asarray(self.s_limit_mva, dtype=float).tolist()
            ],
            "status": str(self.status),
            "pf_attempt": str(self.pf_attempt),
            "pf_repairs": list(self.pf_repairs),
            "gen_dispatch_mw_by_name": [
                (str(k), float(v)) for k, v in self.gen_dispatch_mw_by_name
            ],
        }
