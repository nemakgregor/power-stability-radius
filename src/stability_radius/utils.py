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
from typing import Any, Optional

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


def setup_logging(config: Any) -> str:
    """
    Configure application logging and create a per-run output directory.

    Expected config structure (Hydra/OmegaConf):
      - config.paths.runs_dir: str
      - config.settings.logging.level_console: str (e.g., "INFO")
      - config.settings.logging.level_file: str (e.g., "DEBUG")

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
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    runs_dir = Path(str(config.paths.runs_dir))
    if not runs_dir.is_absolute():
        base = _try_get_original_cwd() or os.getcwd()
        runs_dir = Path(base) / runs_dir

    run_dir = runs_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    level_console = str(config.settings.logging.level_console).upper()
    level_file = str(config.settings.logging.level_file).upper()

    # `force=True` prevents duplicate handlers if setup_logging is called multiple times.
    logging.basicConfig(
        level=logging.DEBUG,  # root at DEBUG; handlers control what is emitted
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(run_dir / "log_debug.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    # Apply levels to handlers deterministically.
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(getattr(logging, level_file, logging.DEBUG))
        elif isinstance(handler, logging.StreamHandler):
            handler.setLevel(getattr(logging, level_console, logging.INFO))

    logger = logging.getLogger(__name__)
    logger.info("New run: %s", timestamp)
    logger.info("Run directory: %s", str(run_dir))
    return str(run_dir)
