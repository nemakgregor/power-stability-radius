"""
Radii computation subpackage.

Public API (restricted)
-----------------------
The project exposes only the core, stable functions as the public API surface.
More specialized DC post-processing (metric / N-1) remains available via explicit
module imports, but is intentionally NOT re-exported here to avoid accidental
coupling and "import *" surface bloat.

Exports
-------
- compute_l2_radius         (DC L2 radii, balanced disturbances)
- compute_ac_l2_radius      (AC L2 radii around AC PF base point)
- compute_sigma_radius      (DC Gaussian post-processing: sigma-radius, overload probability)
- compute_ac_sigma_radius   (AC sigma-radius using precomputed adjoint h-vectors)
- compute_ac_metric_radius  (AC metric radius under arbitrary SPD weight matrix)
"""

from __future__ import annotations

from .ac_l2 import compute_ac_l2_radius
from .ac_metric_radius import compute_ac_metric_radius
from .ac_sigma_radius import compute_ac_sigma_radius
from .l2 import compute_l2_radius
from .probabilistic import compute_sigma_radius

__all__ = [
    "compute_l2_radius",
    "compute_ac_l2_radius",
    "compute_sigma_radius",
    "compute_ac_sigma_radius",
    "compute_ac_metric_radius",
]
