"""Parse UnitCommitment.jl JSON instances and extract per-bus injection σ."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _load_json(file_path: str | Path) -> dict[str, Any]:
    """Read and parse a UC.jl JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() != ".json":
        raise ValueError(f"Expected a .json file, got: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _as_list(value: Any) -> list[float]:
    """Coerce a scalar or list to a list of floats."""
    if isinstance(value, list):
        return [float(v) for v in value]
    return [float(value)]


def _build_bus_mapping(
    bus_names: list[str],
    explicit_mapping: dict[str, int] | None,
) -> dict[str, int]:
    """Return a mapping from UC.jl bus name to integer index.

    If *explicit_mapping* is given it is validated and returned.
    Otherwise a mapping is inferred by sorting bus names and assigning
    consecutive indices starting from 0.
    """
    if explicit_mapping is not None:
        missing = set(bus_names) - set(explicit_mapping)
        if missing:
            raise ValueError(
                f"Explicit bus mapping is missing entries for: {sorted(missing)}"
            )
        return dict(explicit_mapping)

    sorted_names = sorted(bus_names)
    return {name: idx for idx, name in enumerate(sorted_names)}


def load_sigma(
    file_path: str | Path,
    *,
    bus_mapping: dict[str, int] | None = None,
    power_factor: float = 0.9,
) -> dict[str, Any]:
    """Load a UnitCommitment.jl JSON instance and extract per-bus σ.

    Parameters
    ----------
    file_path:
        Path to the UC.jl JSON file.
    bus_mapping:
        Optional explicit ``{"b1": 0, "b2": 1, ...}`` mapping. When
        *None* the mapping is inferred from sorted bus names.
    power_factor:
        Power factor used to estimate reactive-power σ from active-power
        σ via ``σ_Q = σ_P * tan(arccos(pf))``.  Default ``0.9``.

    Returns
    -------
    dict with keys:
        sigma_p_mw : np.ndarray (n_bus,)
        sigma_q_mvar : np.ndarray (n_bus,)
        n_timesteps : int
        bus_mapping : dict[str, int]
        metadata : dict
    """
    path = Path(file_path)
    data = _load_json(path)

    buses: dict[str, Any] = data.get("Buses", {})
    generators: dict[str, Any] = data.get("Generators", {})

    if not buses:
        raise ValueError(f"No 'Buses' section found in {path}")

    bus_names = list(buses.keys())
    mapping = _build_bus_mapping(bus_names, bus_mapping)
    n_bus = len(bus_names)

    # --- σ_load per bus (std of the load time series) ---
    sigma_load = np.zeros(n_bus, dtype=float)
    n_timesteps = 0

    for bname, bdata in buses.items():
        load_ts = _as_list(bdata.get("Load (MW)", 0.0))
        n_timesteps = max(n_timesteps, len(load_ts))
        idx = mapping[bname]
        if len(load_ts) > 1:
            sigma_load[idx] = float(np.std(load_ts, ddof=0))

    # --- σ_gen per bus (std of time-varying generator capacity) ---
    sigma_gen = np.zeros(n_bus, dtype=float)

    for _gname, gdata in generators.items():
        gen_bus = gdata.get("Bus")
        if gen_bus is None or gen_bus not in mapping:
            continue
        idx = mapping[gen_bus]

        max_power = gdata.get("Max power (MW)")
        if max_power is None:
            continue
        capacity_ts = _as_list(max_power)
        n_timesteps = max(n_timesteps, len(capacity_ts))
        if len(capacity_ts) > 1:
            gen_std = float(np.std(capacity_ts, ddof=0))
            # Accumulate variance (multiple generators may sit on the same bus)
            sigma_gen[idx] = math.sqrt(sigma_gen[idx] ** 2 + gen_std**2)

    # --- total σ_P ---
    sigma_p = np.sqrt(sigma_load**2 + sigma_gen**2)

    # --- σ_Q estimate ---
    tan_phi = math.tan(math.acos(power_factor))
    sigma_q = sigma_p * tan_phi

    logger.debug(
        "UC.jl parsed: %d buses, %d generators, %d timesteps",
        n_bus,
        len(generators),
        n_timesteps,
    )

    return {
        "sigma_p_mw": sigma_p,
        "sigma_q_mvar": sigma_q,
        "n_timesteps": n_timesteps,
        "bus_mapping": mapping,
        "metadata": {
            "source": str(path),
            "n_buses": n_bus,
            "n_generators": len(generators),
            "power_factor": power_factor,
        },
    }
