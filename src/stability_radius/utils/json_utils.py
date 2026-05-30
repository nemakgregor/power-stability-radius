from __future__ import annotations

"""JSON helpers for numpy-heavy result structures."""

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json_object(path: str | Path) -> dict[str, Any]:
    """Load a JSON file and require the top-level value to be an object."""
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {p}, got {type(obj)}")
    return obj


def result_meta(results: dict[str, Any]) -> dict[str, Any]:
    """Return the `__meta__` object from a results dictionary when present."""
    meta = results.get("__meta__")
    return meta if isinstance(meta, dict) else {}


def numpy_to_builtin(obj: object) -> object:
    """Convert numpy scalars/arrays to plain Python objects for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class NumpyJSONEncoder(json.JSONEncoder):
    """`json.JSONEncoder` variant that understands numpy objects."""

    def default(self, obj: Any) -> Any:
        """Execute the documented operation."""
        try:
            return numpy_to_builtin(obj)
        except TypeError:
            return super().default(obj)
