from __future__ import annotations

"""
Project utilities.

Logging design (deterministic, project-only)
--------------------------------------------
- All project loggers are under the "stability_radius" namespace.
- We attach handlers ONLY to "stability_radius" logger (not to root), so third-party
  logs do not pollute outputs.
- A dedicated "stability_radius.fileonly" logger is configured with the file handler only
  to support large outputs (e.g., ASCII tables) without printing them to console.
"""

import logging
import os
import shutil
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from stability_radius.config import LoggingConfig

__all__ = ["log_stage", "setup_logging"]

_LOGGER_ROOT_NAME = "stability_radius"
_LOGGER_FILE_ONLY_NAME = "stability_radius.fileonly"


def _level_from_str(level: str) -> int:
    """Convert 'INFO'/'DEBUG'/... to logging level integer."""
    lvl = getattr(logging, str(level).upper(), None)
    if not isinstance(lvl, int):
        raise ValueError(f"Invalid logging level: {level!r}")
    return int(lvl)


def _close_and_clear_handlers(lg: logging.Logger) -> None:
    """Close existing handlers to avoid file descriptor leaks across repeated setup_logging() calls."""
    for h in list(lg.handlers):
        try:
            h.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass
    lg.handlers.clear()


def _make_unique_run_dir(runs_dir: Path, *, prefix: str) -> Path:
    """
    Create a unique run directory under `runs_dir`.

    Deterministic collision handling:
    - if <prefix>/ exists, append _01, _02, ...
    """
    candidate = runs_dir / prefix
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    i = 1
    while True:
        suffixed = runs_dir / f"{prefix}_{i:02d}"
        if not suffixed.exists():
            suffixed.mkdir(parents=True, exist_ok=False)
            return suffixed
        i += 1


@contextmanager
def log_stage(stage_logger: logging.Logger, stage_name: str):
    """
    Log a workflow stage boundary with duration.

    Format:
        ==> [START] <name>
        <== [END] <name> (time taken: X.XXX sec)

    On exception:
        <!! [FAIL] <name> (time taken: X.XXX sec)  + traceback
    """
    t0 = time.perf_counter()
    stage_logger.info("==> [START] %s", str(stage_name))
    try:
        yield
    except Exception:
        dt = time.perf_counter() - t0
        stage_logger.exception(
            "<!! [FAIL] %s (time taken: %.3f sec)", str(stage_name), float(dt)
        )
        raise
    else:
        dt = time.perf_counter() - t0
        stage_logger.info(
            "<== [END] %s (time taken: %.3f sec)", str(stage_name), float(dt)
        )


def setup_logging(cfg: LoggingConfig) -> str:
    """
    Configure project logging and create a per-run output directory.

    Parameters
    ----------
    cfg:
        LoggingConfig with:
          - runs_dir
          - level_console
          - level_file
          - run_dir_mode: "timestamp" | "overwrite"
          - run_name: used only for "overwrite"

    Returns
    -------
    str
        Absolute path to the created run directory.
    """
    runs_dir = Path(str(cfg.runs_dir))
    if not runs_dir.is_absolute():
        runs_dir = (Path(os.getcwd()) / runs_dir).resolve()

    mode = str(getattr(cfg, "run_dir_mode", "timestamp")).strip().lower()
    if mode not in {"timestamp", "overwrite"}:
        raise ValueError("run_dir_mode must be 'timestamp' or 'overwrite'.")

    if mode == "overwrite":
        run_name = str(getattr(cfg, "run_name", "latest")).strip() or "latest"
        run_dir = runs_dir / run_name
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        run_dir = _make_unique_run_dir(runs_dir, prefix=timestamp)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setLevel(_level_from_str(cfg.level_file))
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(_level_from_str(cfg.level_console))
    console_handler.setFormatter(fmt)

    # Clear root handlers to prevent accidental third-party logging to console/file.
    root = logging.getLogger()
    _close_and_clear_handlers(root)
    root.setLevel(logging.WARNING)

    project_logger = logging.getLogger(_LOGGER_ROOT_NAME)
    _close_and_clear_handlers(project_logger)
    project_logger.setLevel(logging.DEBUG)
    project_logger.propagate = False
    project_logger.addHandler(file_handler)
    project_logger.addHandler(console_handler)

    file_only_logger = logging.getLogger(_LOGGER_FILE_ONLY_NAME)
    _close_and_clear_handlers(file_only_logger)
    file_only_logger.setLevel(logging.DEBUG)
    file_only_logger.propagate = False
    file_only_logger.addHandler(file_handler)

    project_logger.info("New run: %s", run_dir.name)
    project_logger.info("Run directory: %s", str(run_dir))
    project_logger.info("Log file: %s", str(run_dir / "run.log"))
    return str(run_dir.resolve())
