from __future__ import annotations

"""JSON helpers for numpy-heavy result structures."""

import json
from typing import Any

import numpy as np


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
