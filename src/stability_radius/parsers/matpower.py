from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_EPS_TAP = 1e-12


def _strip_matpower_comments(text: str) -> str:
    """Remove MATPOWER comments (% ... end-of-line)."""
    return re.sub(r"%.*$", "", text, flags=re.MULTILINE)


def _extract_scalar(text: str, name: str) -> str:
    """Extract `mpc.<name> = <value>;` scalar assignment from MATPOWER case text."""
    pattern = re.compile(
        rf"\bmpc\.{re.escape(name)}\s*=\s*([^;]+?)\s*;", flags=re.MULTILINE
    )
    m = pattern.search(text)
    if not m:
        raise ValueError(f"Could not find scalar 'mpc.{name}' in MATPOWER case.")
    return m.group(1).strip()


def _extract_matrix(text: str, name: str) -> np.ndarray:
    """Extract numeric matrix `mpc.<name> = [ ... ];` as 2D float array."""
    pattern = re.compile(
        rf"\bmpc\.{re.escape(name)}\s*=\s*\[(.*?)\]\s*;",
        flags=re.MULTILINE | re.DOTALL,
    )
    m = pattern.search(text)
    if not m:
        raise ValueError(f"Could not find matrix 'mpc.{name}' in MATPOWER case.")

    body = m.group(1)
    body = body.replace("...", " ").replace(",", " ")
    body = re.sub(r"\s+", " ", body).strip()

    if not body:
        return np.zeros((0, 0), dtype=float)

    rows_raw = [r.strip() for r in body.split(";") if r.strip()]
    rows: list[list[float]] = []
    for r in rows_raw:
        parts = [p for p in r.split(" ") if p]
        try:
            rows.append([float(x) for x in parts])
        except ValueError as e:
            raise ValueError(f"Failed to parse numeric row in mpc.{name}: {r!r}") from e

    if not rows:
        return np.zeros((0, 0), dtype=float)

    ncols = len(rows[0])
    if any(len(r) != ncols for r in rows):
        raise ValueError(
            f"Inconsistent row lengths in mpc.{name} matrix (expected {ncols})."
        )

    return np.asarray(rows, dtype=float)


def _parse_matpower_m_file_to_ppc(path: Path) -> dict[str, Any]:
    """
    Parse a MATPOWER `.m` case file into a minimal PPC dict.

    Fields:
      - version
      - baseMVA
      - bus
      - gen
      - branch
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    text = _strip_matpower_comments(text)

    version_raw = _extract_scalar(text, "version")
    version = version_raw.strip().strip("'").strip('"')

    base_mva_raw = _extract_scalar(text, "baseMVA")
    try:
        base_mva = float(base_mva_raw)
    except ValueError as e:
        raise ValueError(
            f"Failed to parse mpc.baseMVA={base_mva_raw!r} as float."
        ) from e

    bus = _extract_matrix(text, "bus")
    gen = _extract_matrix(text, "gen")
    branch = _extract_matrix(text, "branch")

    if bus.size == 0 or branch.size == 0:
        raise ValueError(
            "MATPOWER case must contain non-empty 'bus' and 'branch' matrices."
        )

    return {
        "version": version,
        "baseMVA": base_mva,
        "bus": bus,
        "gen": gen,
        "branch": branch,
    }


def _from_ppc_to_pandapower(ppc: dict[str, Any], f_hz: float):
    """
    Convert a PPC dict to a pandapower network.

    This project targets a single modern pandapower API:
      pandapower.converter.pypower.from_ppc.from_ppc(ppc, f_hz=..., validate_conversion=False)
    """
    from pandapower.converter.pypower.from_ppc import from_ppc

    return from_ppc(ppc, f_hz=float(f_hz), validate_conversion=False)


def _attach_matpower_rateA_to_net_lines(*, ppc: dict[str, Any], net: Any) -> None:
    """
    Ensure MATPOWER/PGLib thermal limits (branch.rateA) are available on pandapower net.line.

    Why
    ---
    pandapower's from_ppc() does not always preserve MATPOWER branch ratings into
    net.line columns. For PGLib-OPF workflows we need rateA to build consistent
    thermal constraints (OPF + radii + MC).

    Mapping rule (explicit, MATPOWER-consistent)
    -------------------------------------------
    We treat PPC branches with TAP == 0 (or NaN) as "lines" (no transformer).
    These are mapped in-order to net.line entries (also in-order by index).

    If counts mismatch, we log a warning and do NOT attach values (fallback remains max_i_ka-based
    if available, otherwise radii pipeline will fail with an explicit error).
    """
    if not hasattr(net, "line") or net.line is None or len(net.line) == 0:
        return

    branch = np.asarray(ppc.get("branch", np.zeros((0, 0))), dtype=float)
    if branch.ndim != 2 or branch.shape[0] == 0 or branch.shape[1] <= 5:
        return

    rateA = np.asarray(branch[:, 5], dtype=float)  # RATE_A column (MVA)
    tap = (
        np.asarray(branch[:, 8], dtype=float)
        if branch.shape[1] > 8
        else np.zeros(branch.shape[0], dtype=float)
    )

    mask_line = (~np.isfinite(tap)) | (np.abs(tap) <= _EPS_TAP)
    rateA_lines = rateA[mask_line]

    if int(rateA_lines.size) != int(len(net.line)):
        logger.warning(
            "MATPOWER->pandapower: cannot map branch.rateA to net.line. "
            "Expected line-like branches count=%d (tap==0), but net.line count=%d. "
            "Will rely on other rating sources (e.g., max_i_ka) if present.",
            int(rateA_lines.size),
            int(len(net.line)),
        )
        return

    idx = [int(x) for x in sorted(net.line.index)]
    net.line.loc[idx, "rateA"] = rateA_lines
    logger.debug(
        "Attached MATPOWER branch.rateA into net.line['rateA'] for %d lines.", len(idx)
    )


def load_network(file_path: str | Path, f_hz: float = 50.0):
    """
    Load a MATPOWER/PGLib `.m` case file into a pandapower network.

    Deterministic policy
    --------------------
    - Uses the internal deterministic `.m` parser from this repository.
    - Converts PPC -> pandapower via pandapower's pypower converter.
    - No optional fallbacks.

    Post-processing (important)
    ---------------------------
    - Propagates MATPOWER branch rateA into pandapower net.line['rateA'] where possible,
      to ensure consistent thermal limit extraction for OPF/radii pipelines.

    Raises
    ------
    FileNotFoundError:
        If file does not exist.
    ValueError:
        If extension is not `.m`.
    RuntimeError:
        If parsing or conversion fails.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.suffix.lower() != ".m":
        raise ValueError(f"Only MATPOWER/PGLib .m files are supported. Got: {path}")

    try:
        ppc = _parse_matpower_m_file_to_ppc(path)
        net = _from_ppc_to_pandapower(ppc, f_hz=float(f_hz))

        # Ensure MATPOWER thermal ratings are available for downstream limit extraction.
        _attach_matpower_rateA_to_net_lines(ppc=ppc, net=net)

        logger.debug("Network loaded: %d buses, %d lines", len(net.bus), len(net.line))
        return net
    except Exception as e:
        logger.exception("Failed to load MATPOWER case: %s", str(path))
        raise RuntimeError(f"Failed to load MATPOWER case: {path}") from e
