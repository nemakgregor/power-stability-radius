"""
AC linearization helpers.

This subpackage provides:
- Building an AC PF Jacobian operator around a base point (V, theta)
- Fast adjoint-based sensitivities for line-flow constraints
"""

from __future__ import annotations

from .ac_model import ACOperator, build_ac_operator

__all__ = ["ACOperator", "build_ac_operator"]
