"""Geometry helpers for disturbance spaces and dual norms."""

from .balanced import (
    BlockSpec,
    dual_norm_l2_balanced,
    dual_norm_l2_balanced_from_block_vectors,
    dual_norm_l2_balanced_rows,
    make_ac_block_specs,
    project_dual_balanced,
    project_dual_balanced_rows,
    worst_case_l2_direction,
)

__all__ = [
    "BlockSpec",
    "dual_norm_l2_balanced",
    "dual_norm_l2_balanced_from_block_vectors",
    "dual_norm_l2_balanced_rows",
    "make_ac_block_specs",
    "project_dual_balanced",
    "project_dual_balanced_rows",
    "worst_case_l2_direction",
]
