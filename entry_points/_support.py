from __future__ import annotations

"""Shared helpers for scripts under `entry_points/`."""

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


def bootstrap_repo_paths() -> None:
    """Ensure `src/` and the repository root are importable for entry-point scripts."""
    for path in (SRC_DIR, REPO_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def run_entrypoint(module_name: str, func_name: str = "main") -> int:
    """Import `module_name.func_name`, execute it, and normalize the exit code."""
    bootstrap_repo_paths()
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    result = func()
    return 0 if result is None else int(result)

