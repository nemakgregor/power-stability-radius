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
import re
import shutil
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from stability_radius.config import DEFAULT_LOGGING, LoggingConfig

__all__ = [
    "ARTIFACTS_ROOT_NAME",
    "create_module_output_dir",
    "log_stage",
    "resolve_artifacts_root",
    "setup_logging",
    "setup_output_dir_logging",
]

_LOGGER_ROOT_NAME = "stability_radius"
_LOGGER_FILE_ONLY_NAME = "stability_radius.fileonly"
ARTIFACTS_ROOT_NAME = DEFAULT_LOGGING.runs_dir


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


def _resolve_base_dir(path_value: str | Path) -> Path:
    """Expand `~` and resolve a potentially-relative base directory."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (Path(os.getcwd()) / path).resolve()
    else:
        path = path.resolve()
    return path


def _sanitize_path_component(value: str) -> str:
    """Normalize a user/module name into a safe directory component."""
    raw = str(value).strip()
    if not raw:
        return "general"
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", raw)
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = cleaned.strip("._")
    return cleaned or "general"


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return True iff `path` is located under `parent`."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


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


def create_module_output_dir(
    *,
    module_name: str,
    runs_dir: str | Path | None = None,
    requested_output_dir: str | Path | None = None,
) -> Path:
    """
    Resolve a module-specific artifact directory under the configured artifact root.

    Rules:
    - default: `<artifacts_root>/<module_name>`
    - if `requested_output_dir` already points inside `<artifacts_root>/`, keep it
    - otherwise normalize it to `<artifacts_root>/<module_name>/<basename>`
    """
    runs_root = _resolve_base_dir(
        DEFAULT_LOGGING.runs_dir if runs_dir is None else runs_dir
    )
    module_dir = runs_root / _sanitize_path_component(module_name)

    requested_raw = "" if requested_output_dir is None else str(requested_output_dir).strip()
    if not requested_raw:
        module_dir.mkdir(parents=True, exist_ok=True)
        return module_dir.resolve()

    requested = Path(requested_raw).expanduser()
    resolved_requested = (
        requested.resolve()
        if requested.is_absolute()
        else (Path(os.getcwd()) / requested).resolve()
    )
    if _is_relative_to(resolved_requested, runs_root):
        resolved_requested.mkdir(parents=True, exist_ok=True)
        return resolved_requested.resolve()

    target = module_dir / _sanitize_path_component(requested.name or module_name)
    target.mkdir(parents=True, exist_ok=True)
    return target.resolve()


def resolve_artifacts_root(
    cfg: Mapping[str, object] | None = None,
    *,
    runs_dir: str | Path | None = None,
) -> Path:
    """
    Resolve the shared artifact root for an entry point.

    Priority:
    1. explicit `runs_dir` argument
    2. config mapping key `logging.runs_dir`
    3. config mapping key `artifacts_root`
    4. project default (`DEFAULT_LOGGING.runs_dir`)
    """
    if runs_dir is not None and str(runs_dir).strip():
        return _resolve_base_dir(runs_dir)

    if cfg is not None:
        logging_cfg = cfg.get("logging")
        if isinstance(logging_cfg, Mapping):
            logging_runs_dir = logging_cfg.get("runs_dir")
            if logging_runs_dir is not None and str(logging_runs_dir).strip():
                return _resolve_base_dir(str(logging_runs_dir))

        artifacts_root = cfg.get("artifacts_root")
        if artifacts_root is not None and str(artifacts_root).strip():
            return _resolve_base_dir(str(artifacts_root))

    return _resolve_base_dir(DEFAULT_LOGGING.runs_dir)


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
          - runs_dir (supports "~", relative paths are resolved against CWD)
          - module_name (top-level module/pipeline group under runs_dir)
          - level_console
          - level_file
          - run_dir_mode: "timestamp" | "overwrite"
          - run_name: used only for "overwrite"

    Returns
    -------
    str
        Absolute path to the created run directory.
    """
    runs_dir_raw = str(getattr(cfg, "runs_dir", "")).strip()
    if not runs_dir_raw:
        raise ValueError("runs_dir must be a non-empty path.")

    runs_root = _resolve_base_dir(runs_dir_raw)
    module_dir = runs_root / _sanitize_path_component(getattr(cfg, "module_name", "general"))

    mode = str(getattr(cfg, "run_dir_mode", "timestamp")).strip().lower()
    if mode not in {"timestamp", "overwrite"}:
        raise ValueError("run_dir_mode must be 'timestamp' or 'overwrite'.")

    log_filename = Path(
        str(getattr(cfg, "log_filename", "debug.log")).strip() or "debug.log"
    ).name

    if mode == "overwrite":
        run_name = str(getattr(cfg, "run_name", "latest")).strip() or "latest"
        run_dir = module_dir / _sanitize_path_component(run_name)
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        run_dir = _make_unique_run_dir(module_dir, prefix=timestamp)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.FileHandler(run_dir / log_filename, encoding="utf-8")
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
    project_logger.info("Runs root: %s", str(runs_root))
    project_logger.info("Module directory: %s", str(module_dir))
    project_logger.info("Run directory: %s", str(run_dir))
    project_logger.info("Run directory mode: %s", mode)
    if mode == "overwrite":
        project_logger.info("Run name: %s", str(getattr(cfg, "run_name", "latest")))
    project_logger.info("Log file: %s", str(run_dir / log_filename))
    return str(run_dir.resolve())


def setup_output_dir_logging(
    output_dir: str | Path,
    *,
    level_console: str = "INFO",
    level_file: str = "DEBUG",
    log_filename: str = "debug.log",
    quiet_loggers: Sequence[str] = ("pandapower", "numba", "urllib3", "matplotlib"),
) -> Path:
    """
    Configure console + file logging for scripts that write into a fixed output directory.

    Unlike `setup_logging()`, this helper does not create a timestamped run directory.
    It writes directly into `output_dir/log_filename`.
    """
    target_dir = _resolve_base_dir(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    log_path = target_dir / (Path(log_filename).name or "debug.log")

    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(_level_from_str(level_console))
    console_handler.setFormatter(fmt)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(_level_from_str(level_file))
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    _close_and_clear_handlers(root)
    root.setLevel(logging.DEBUG)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    project_logger = logging.getLogger(_LOGGER_ROOT_NAME)
    _close_and_clear_handlers(project_logger)
    project_logger.setLevel(logging.DEBUG)
    project_logger.propagate = True

    file_only_logger = logging.getLogger(_LOGGER_FILE_ONLY_NAME)
    _close_and_clear_handlers(file_only_logger)
    file_only_logger.setLevel(logging.DEBUG)
    file_only_logger.propagate = True

    for name in quiet_loggers:
        logging.getLogger(str(name)).setLevel(logging.WARNING)

    return log_path.resolve()
