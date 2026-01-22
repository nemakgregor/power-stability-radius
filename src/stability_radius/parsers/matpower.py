from __future__ import annotations

import inspect
import logging
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
from pandapower.converter.matpower import from_mpc

logger = logging.getLogger(__name__)


def _strip_matpower_comments(text: str) -> str:
    """
    Remove MATPOWER comments from text.

    MATPOWER uses `%` for comments. This function removes everything from `%` to
    end of line.
    """
    return re.sub(r"%.*$", "", text, flags=re.MULTILINE)


def _extract_scalar(text: str, name: str) -> str:
    """
    Extract a scalar assignment like `mpc.baseMVA = 100;` from MATPOWER case text.

    Parameters
    ----------
    text:
        MATPOWER case file contents (comments already stripped is recommended).
    name:
        Field name under `mpc.<name>`.

    Returns
    -------
    str
        Raw scalar value as string (caller converts to numeric if needed).

    Raises
    ------
    ValueError
        If the scalar cannot be found.
    """
    pattern = re.compile(
        rf"\bmpc\.{re.escape(name)}\s*=\s*([^;]+?)\s*;", flags=re.MULTILINE
    )
    m = pattern.search(text)
    if not m:
        raise ValueError(f"Could not find scalar 'mpc.{name}' in MATPOWER case.")
    return m.group(1).strip()


def _extract_matrix(text: str, name: str) -> np.ndarray:
    """
    Extract a numeric matrix like `mpc.bus = [ ... ];` from MATPOWER case text.

    Parameters
    ----------
    text:
        MATPOWER case file contents (comments already stripped is recommended).
    name:
        One of `bus`, `gen`, `branch`, etc.

    Returns
    -------
    np.ndarray
        2D float array.

    Raises
    ------
    ValueError
        If the matrix cannot be found or parsed.
    """
    pattern = re.compile(
        rf"\bmpc\.{re.escape(name)}\s*=\s*\[(.*?)\]\s*;",
        flags=re.MULTILINE | re.DOTALL,
    )
    m = pattern.search(text)
    if not m:
        raise ValueError(f"Could not find matrix 'mpc.{name}' in MATPOWER case.")

    body = m.group(1)
    body = body.replace("...", " ")
    body = body.replace(",", " ")
    body = re.sub(r"\s+", " ", body).strip()

    if not body:
        return np.zeros((0, 0), dtype=float)

    rows_raw = [r.strip() for r in body.split(";") if r.strip()]
    rows = []
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


def _parse_matpower_m_file_to_ppc(path: Path) -> Dict[str, Any]:
    """
    Parse a MATPOWER `.m` case file into a pypower-like PPC dict.

    This is a fallback for environments where pandapower's `.m` converter requires
    the optional `matpowercaseframes` dependency.

    Notes
    -----
    This parser is intentionally minimal: it reads baseMVA, version, bus, gen, branch.
    It is sufficient for standard IEEE MATPOWER cases.
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


def _suppress_known_pandapower_pandas_warnings() -> None:
    """
    Suppress known noisy FutureWarnings emitted inside pandapower converters.

    These warnings come from pandapower internals interacting with newer pandas.
    They are not actionable within this project without pinning/upgrading pandapower.
    """
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r"The behavior of DataFrame concatenation with empty or all-NA entries is deprecated\..*",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r"Setting an item of incompatible dtype is deprecated and will raise an error in a future version of pandas\..*",
    )


def _from_ppc_to_pandapower(ppc: Dict[str, Any], f_hz: float):
    """
    Convert a PPC dict to a pandapower network.

    The signature of `pandapower.converter.pypower.from_ppc` differs across pandapower
    versions; this helper passes only supported kwargs.
    """
    from pandapower.converter.pypower.from_ppc import (
        from_ppc,
    )  # local import on purpose

    sig = inspect.signature(from_ppc)
    kwargs: Dict[str, Any] = {}

    if "f_hz" in sig.parameters:
        kwargs["f_hz"] = f_hz
    if "validate_conversion" in sig.parameters:
        kwargs["validate_conversion"] = False

    with warnings.catch_warnings():
        _suppress_known_pandapower_pandas_warnings()
        return from_ppc(ppc, **kwargs)


def _from_mpc_to_pandapower(path: Path, f_hz: float):
    """Call pandapower's from_mpc with version-tolerant kwargs."""
    sig = inspect.signature(from_mpc)
    kwargs: Dict[str, Any] = {}
    if "f_hz" in sig.parameters:
        kwargs["f_hz"] = f_hz
    if "validate_conversion" in sig.parameters:
        kwargs["validate_conversion"] = False

    with warnings.catch_warnings():
        _suppress_known_pandapower_pandas_warnings()
        return from_mpc(str(path), **kwargs)


def load_network(file_path: Union[str, Path], f_hz: float = 50.0):
    """
    Load a MATPOWER `.m`/`.mat` case file into a pandapower network.

    Behavior
    --------
    1) Tries `pandapower.converter.matpower.from_mpc`.
    2) If pandapower raises `NotImplementedError` due to missing optional
       dependency `matpowercaseframes`, falls back to an internal `.m` parser
       and converts via `pandapower.converter.pypower.from_ppc`.

    Logging
    -------
    Uses DEBUG-level logs for normal progress to keep CLI output clean. Errors are
    still logged with exception details.

    Parameters
    ----------
    file_path:
        Path to the MATPOWER case file.
    f_hz:
        Network frequency in Hz.

    Returns
    -------
    pandapowerNet
        Converted pandapower network.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    RuntimeError
        If conversion fails.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    try:
        net = _from_mpc_to_pandapower(path, f_hz=f_hz)
        logger.debug("Network loaded: %d buses, %d lines", len(net.bus), len(net.line))
        return net
    except NotImplementedError as e:
        # pandapower's from_mpc raises this when matpowercaseframes is not installed.
        logger.debug(
            "pandapower from_mpc() is not available (%s). Using internal MATPOWER .m parser fallback.",
            str(e),
        )
        try:
            ppc = _parse_matpower_m_file_to_ppc(path)
            net = _from_ppc_to_pandapower(ppc, f_hz=f_hz)
            logger.debug(
                "Network loaded (fallback parser): %d buses, %d lines",
                len(net.bus),
                len(net.line),
            )
            return net
        except Exception as e2:
            logger.exception("Fallback MATPOWER .m parsing failed for %s", str(path))
            raise RuntimeError(
                f"Failed to load MATPOWER case (fallback parser): {path}"
            ) from e2
    except Exception as e:
        logger.exception("Failed to load MATPOWER case: %s", str(path))
        raise RuntimeError(f"Failed to load MATPOWER case: {path}") from e
