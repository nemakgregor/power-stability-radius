"""
stability_radius package.

The repository uses a `src/` layout. Library code lives under `src/stability_radius`.

Public API
----------
- `compute_results_for_case`: main deterministic single-case pipeline (OPF -> DC model -> radii).
"""

from __future__ import annotations

from .workflows import compute_results_for_case

__all__ = ["__version__", "compute_results_for_case"]

__version__ = "0.1.0"
