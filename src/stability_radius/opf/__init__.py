"""
OPF helpers.

Project policy
--------------
The base point is solved ONLY via:
- PyPSA DC OPF + HiGHS

This subpackage should remain lightweight and be imported only when OPF is needed.
"""

from __future__ import annotations

from .pypsa_opf import solve_dc_opf_base_flows_from_pandapower

__all__ = ["solve_dc_opf_base_flows_from_pandapower"]
