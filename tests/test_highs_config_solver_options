from __future__ import annotations


def test_highs_solver_options_include_scaling_and_tolerances() -> None:
    from stability_radius.config import HiGHSConfig

    cfg = HiGHSConfig(threads=1, random_seed=42)
    opts = cfg.solver_options()

    assert opts["threads"] == 1
    assert opts["random_seed"] == 42

    # Stability-critical defaults (must be present to avoid silent regressions).
    assert opts["user_objective_scale"] == -1
    assert opts["user_bound_scale"] == -10

    assert float(opts["primal_feasibility_tolerance"]) <= 1e-9
    assert float(opts["dual_feasibility_tolerance"]) <= 1e-9
