# Repository Structure

This document tracks the current repository layout as it exists in code, not as a historical snapshot. If you add, remove, or rename an entry point or package directory, update this file and [entry_points.md](entry_points.md) in the same change.

## Top Level

```text
power-stability-radius/
|-- entry_points/          Runnable scripts and experiment front doors
|-- src/stability_radius/  Reusable library code
|-- tests/                 Pytest suite
|-- conf/                  Main YAML configuration chain
|-- docs/                  Versioned documentation
|-- experiments/           Experiment-specific YAML configs
|-- data/                  Input cases and external datasets
|-- .github/workflows/     GitHub Actions configuration
|-- README.md              Repository landing page
|-- UNITS_CONTRACT.md      Detailed units and schema contract
|-- pyproject.toml         Poetry, pytest, and Ruff configuration
`-- poetry.lock            Locked dependencies
```

## `entry_points/`

`entry_points/` contains every runnable module. The canonical inventory lives in [entry_points.md](entry_points.md), but the directory is organized into three groups:

- Main CLI: `power_stability_radius.py`
- Experiment and analysis workflows: `run_pglib_sweep.py`, `run_sigma_radius.py`, `run_worst_case_verify.py`, `run_scalability.py`, `metrics_analysis.py`, `n1_stability_demo.py`
- Reporting and plotting helpers: `table.py`, `collect_results.py`, `plot_radius_distribution.py`, `plot_sigma_vs_time.py`, `plot_worst_case_heatmap.py`

## `src/stability_radius/`

The package is split by responsibility:

- `workflows.py`: top-level orchestration for single-case computation
- `config.py`: defaults and YAML composition helpers
- `ac/`: AC operator construction and Jacobian-based sensitivities
- `dc/`: DC operator construction and PTDF-like sensitivities
- `base_point/`: DC OPF, AC PF, AC FPF, and pandapower integration helpers
- `radii/`: DC and AC radius implementations
- `verification/`: Monte Carlo, worst-case verification, report generation, and status summaries
- `metrics/`: baseline and practical metric computations
- `parsers/`: MATPOWER and UnitCommitment.jl parsers
- `opf/`: OPF and PF helper wrappers shared across workflows
- `utils/`: logging, artifact-directory helpers, download helpers, JSON utilities
- `pp_helpers.py`: small pandapower table helpers used across modules

## `tests/`

The test suite covers:

- Core math and operator logic in `src/stability_radius/`
- CLI and entry-point wiring in `entry_points/`
- Docs-as-code checks for entry-point documentation and CI workflow drift
- Smoke coverage for experiment/reporting helper scripts

Start with [testing_and_ci.md](testing_and_ci.md) for local commands and CI expectations.

## `conf/` and `experiments/configs/`

- `conf/` contains the main composable config chain used by `entry_points/power_stability_radius.py`
- `experiments/configs/` contains standalone experiment configs such as `pglib_sweep.yaml` and sigma-radius case definitions

## Artifacts

Generated outputs are written under `run_artifacts/` by default:

- Main CLI commands create per-run directories such as `run_artifacts/compute/<timestamp>/`
- Standalone scripts usually write under `run_artifacts/<module>/`
- Some scripts normalize user-provided output paths back under `run_artifacts/` to keep artifacts colocated and reproducible
