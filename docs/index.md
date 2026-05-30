# Power Stability Radius Documentation

This directory is the documentation source of truth for the repository. The project uses a docs-as-code workflow: docs live in Git, the main entry points are documented explicitly, and tests verify that the docs index, entry-point inventory, and CI workflow stay in sync.

## Core References

| File | Purpose |
|------|---------|
| [entry_points.md](entry_points.md) | Authoritative reference for every runnable script under `entry_points/` |
| [execution_flow.md](execution_flow.md) | Accurate runtime flow for the main CLI and experiment scripts |
| [repository_structure.md](repository_structure.md) | Current repository map and package ownership |
| [testing_and_ci.md](testing_and_ci.md) | Local verification commands, docs-as-code checks, and GitHub Actions details |
| [configuration.md](configuration.md) | YAML configuration structure and parameter behavior |
| [data_formats.md](data_formats.md) | Input and output file contracts |
| [architecture.md](architecture.md) | Conceptual component boundaries and interaction model |
| [developer_guide.md](developer_guide.md) | Extension guidance for new algorithms, metrics, and parsers |
| [mathematical_foundations.md](mathematical_foundations.md) | Formal derivations and notation |
| [algorithms_and_models.md](algorithms_and_models.md) | Algorithmic details for DC, AC, and verification routines |
| [experiments_and_evaluation.md](experiments_and_evaluation.md) | Benchmarking and experimental framing |
| [metrics.md](metrics.md) | Metric definitions and interpretation |
| [limitations_and_assumptions.md](limitations_and_assumptions.md) | Known modeling and implementation limits |
| [reproducibility_and_failfast.md](reproducibility_and_failfast.md) | Determinism and fail-fast policy |
| [n1_demo.md](n1_demo.md) | Details for the dedicated `n1_stability_demo` workflow |
| [glossary.md](glossary.md) | Terminology and symbols |

## Suggested Reading Paths

For a new contributor:

1. [repository_structure.md](repository_structure.md)
2. [entry_points.md](entry_points.md)
3. [execution_flow.md](execution_flow.md)
4. [testing_and_ci.md](testing_and_ci.md)

For a workflow/debugging task:

1. [entry_points.md](entry_points.md)
2. [execution_flow.md](execution_flow.md)
3. [configuration.md](configuration.md)
4. [data_formats.md](data_formats.md)

For an algorithm or research review:

1. [mathematical_foundations.md](mathematical_foundations.md)
2. [algorithms_and_models.md](algorithms_and_models.md)
3. [experiments_and_evaluation.md](experiments_and_evaluation.md)
4. [limitations_and_assumptions.md](limitations_and_assumptions.md)

## Pipeline Overview

The runtime pipeline is:

1. Load a MATPOWER, PGLib, pandapower, or UnitCommitment.jl-derived input.
2. Build a deterministic base operating point from the case, DC OPF, AC PF, or
   AC FPF according to `compute.base_dispatch`.
3. Compute DC and/or AC certificate rows with stable sorted bus and line
   coordinates.
4. Optionally add sigma, metric, N-1, Monte Carlo, or nonlinear AC replay
   checks.
5. Write `results.json`, optional CSV/NPZ artifacts, and verification reports
   under the configured artifact directory.

## Quick Commands

```bash
poetry install
poetry run python entry_points/power_stability_radius.py --config conf/config.yaml compute --input data/input/pglib_opf_case30_ieee.m --slack-bus 0
poetry run python -m pytest -q
poetry run ruff format --check .
```
