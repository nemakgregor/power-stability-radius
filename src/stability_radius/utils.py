"""
Utility helpers for the project.

Repository layout note
----------------------
This repository contains both:

- `stability_radius/utils.py`
- `stability_radius/utils/download.py`

Normally, `utils.py` would shadow a `utils/` package directory, breaking imports like:
    from stability_radius.utils.download import download_ieee_case

To preserve backward compatibility and keep submodule imports working, this module
exposes a `__path__` that points to the sibling `utils/` directory (if present),
so it can behave as a package for submodule imports.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

# Allow `stability_radius.utils.download` to be importable even though `utils.py` exists.
_utils_dir = Path(__file__).with_suffix("")  # ".../utils" next to ".../utils.py"
if _utils_dir.is_dir():
    __path__ = [str(_utils_dir)]  # type: ignore[attr-defined]


def _try_get_original_cwd() -> Optional[str]:
    """Return Hydra original cwd if running under Hydra; otherwise None."""
    try:
        from hydra.utils import get_original_cwd  # type: ignore

        return str(get_original_cwd())
    except Exception:
        return None


def _make_unique_run_dir(runs_dir: Path, *, prefix: str) -> Path:
    """
    Create a unique run directory under `runs_dir`.

    Uses `prefix` and, if needed, a deterministic numeric suffix to avoid collisions.
    This is important for Hydra multiruns or repeated invocations within one second.
    """
    candidate = runs_dir / prefix
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    # Deterministic suffix to avoid collisions without randomness.
    i = 1
    while True:
        suffixed = runs_dir / f"{prefix}_{i:02d}"
        if not suffixed.exists():
            suffixed.mkdir(parents=True, exist_ok=False)
            return suffixed
        i += 1


class _ThirdPartyNoiseFilter(logging.Filter):
    """
    Filter that reduces noise from selected third-party loggers.

    For matching logger name prefixes, it drops records strictly below `min_level`.
    """

    def __init__(self, *, prefixes: Sequence[str], min_level: int) -> None:
        super().__init__()
        self._prefixes = tuple(prefixes)
        self._min_level = int(min_level)

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 (filter)
        name = record.name or ""
        for p in self._prefixes:
            if name == p or name.startswith(p + "."):
                return record.levelno >= self._min_level
        return True


class _ProjectOnlyFilter(logging.Filter):
    """
    Filter that keeps only logs emitted by this project (excludes third-party libs).

    This is applied to both file and console handlers to keep output clean and
    fulfill the requirement "project logs only".
    """

    def __init__(self, *, prefixes: Sequence[str]) -> None:
        super().__init__()
        self._prefixes = tuple(str(p) for p in prefixes)

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 (filter)
        name = record.name or ""
        for p in self._prefixes:
            if name == p or name.startswith(p + "."):
                return True
        return False


class _DropConsoleOnlyFileRecords(logging.Filter):
    """
    Drop records that are marked as "file-only".

    Any log call can set `extra={"sr_only_file": True}` to ensure it never appears
    in console output (but still goes to the file log).
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 (filter)
        return not bool(getattr(record, "sr_only_file", False))


def setup_logging(config: Any) -> str:
    """
    Configure application logging and create a per-run output directory.

    Expected config structure (Hydra/OmegaConf or SimpleNamespace-like):
      - config.paths.runs_dir: str
      - config.settings.logging.level_console: str (e.g., "INFO")
      - config.settings.logging.level_file: str (e.g., "DEBUG")

    Optional config keys (safe defaults used):
      - config.settings.logging.project_prefixes: list[str]
        (default: ["stability_radius", "power_stability_radius", "verification"])
      - config.settings.logging.third_party_min_level: str (default: "WARNING")
      - config.settings.logging.third_party_prefixes: list[str]
        (default: ["numba", "llvmlite"])

    Behavior
    --------
    - Creates runs/<timestamp>/ directory.
    - Writes logs to runs/<timestamp>/run.log (all project levels, DEBUG+ by default).
    - Console logs are filtered:
        * third-party logs are excluded
        * records marked with extra={"sr_only_file": True} are excluded

    Notes
    -----
    When executed under Hydra, the current working directory is changed to Hydra's
    per-run output directory. To keep `runs/` stable at project root, this function
    attempts to resolve relative paths against Hydra's original working directory.

    Returns
    -------
    str
        The created run directory path.
    """
    # Include microseconds to reduce collision probability in fast runs/multiruns.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")

    runs_dir = Path(str(config.paths.runs_dir))
    if not runs_dir.is_absolute():
        base = _try_get_original_cwd() or os.getcwd()
        runs_dir = Path(base) / runs_dir

    run_dir = _make_unique_run_dir(runs_dir, prefix=timestamp)

    # Levels (with safe defaults)
    level_console = str(
        getattr(config.settings.logging, "level_console", "INFO")
    ).upper()
    level_file = str(getattr(config.settings.logging, "level_file", "DEBUG")).upper()

    third_party_min_level_s = str(
        getattr(config.settings.logging, "third_party_min_level", "WARNING")
    ).upper()
    third_party_min_level = getattr(logging, third_party_min_level_s, logging.WARNING)

    third_party_prefixes = getattr(
        config.settings.logging, "third_party_prefixes", ["numba", "llvmlite"]
    )
    try:
        third_party_prefixes = list(third_party_prefixes)
    except Exception:
        third_party_prefixes = ["numba", "llvmlite"]

    project_prefixes = getattr(
        config.settings.logging,
        "project_prefixes",
        ["stability_radius", "power_stability_radius", "verification"],
    )
    try:
        project_prefixes = list(project_prefixes)
    except Exception:
        project_prefixes = [
            "stability_radius",
            "power_stability_radius",
            "verification",
        ]

    # `force=True` prevents duplicate handlers if setup_logging is called multiple times.
    logging.basicConfig(
        level=logging.DEBUG,  # root at DEBUG; handlers control what is emitted
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(run_dir / "run.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    # Apply levels and filters deterministically.
    noise_filter = _ThirdPartyNoiseFilter(
        prefixes=third_party_prefixes, min_level=third_party_min_level
    )
    project_filter = _ProjectOnlyFilter(prefixes=project_prefixes)
    drop_console_file_only = _DropConsoleOnlyFileRecords()

    for handler in logging.getLogger().handlers:
        # Keep only project logs (no third-party libs).
        handler.addFilter(project_filter)

        # Reduce third-party noise (kept for backward compatibility; mostly redundant
        # because project_filter already drops 3rd party logs).
        handler.addFilter(noise_filter)

        if isinstance(handler, logging.FileHandler):
            handler.setLevel(getattr(logging, level_file, logging.DEBUG))
        elif isinstance(handler, logging.StreamHandler):
            handler.setLevel(getattr(logging, level_console, logging.INFO))
            handler.addFilter(drop_console_file_only)

    # Additionally, hard-set logger levels for noisy libraries (affects propagation).
    for p in third_party_prefixes:
        logging.getLogger(str(p)).setLevel(third_party_min_level)

    logger = logging.getLogger("stability_radius.utils")
    logger.info("New run: %s", run_dir.name)
    logger.info("Run directory: %s", str(run_dir))
    logger.info("Log file: %s", str(run_dir / "run.log"))
    return str(run_dir)
