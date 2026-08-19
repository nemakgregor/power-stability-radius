# Power Stability Radius

Power Stability Radius is a Python toolkit for computing robustness certificates for power-system operating points with respect to line thermal-limit violations. The repository contains one primary CLI, several experiment-oriented entry points, and a pytest suite that keeps the implementation, docs, and CI workflow aligned.

Start with the main CLI: `python entry_points/power_stability_radius.py --config conf/config.yaml <command>`

Main commands:

- `compute` or `demo`: run the deterministic DC and/or AC radius pipeline and write `results.json` plus formatted tables.
- `monte-carlo`: verify an existing `results.json` against sampled perturbations.
- `report`: build a multi-case Markdown verification report from `report.cases` in YAML.
- `table`: format an existing `results.json` for terminal output.

Standalone scripts under `entry_points/` cover focused workflows such as `run_pglib_sweep.py`, `run_sigma_radius.py`, `run_worst_case_verify.py`, `run_scalability.py`, `metrics_analysis.py`, and `n1_stability_demo.py`. Reusable table, aggregation, and plotting helpers live under `src/stability_radius/postprocess/`.

Architecturally, the main script in `entry_points/` is now a thin interface wrapper. CLI orchestration lives under `src/stability_radius/application/`, while shared typed models live under `src/stability_radius/domain/`.

Documentation starts at [docs/index.md](docs/index.md). The most useful references for day-to-day work are:

- [docs/entry_points.md](docs/entry_points.md): authoritative inventory of every runnable script and its artifacts.
- [docs/execution_flow.md](docs/execution_flow.md): how the main CLI and experiment flows execute.
- [docs/repository_structure.md](docs/repository_structure.md): where package code, configs, tests, and reports live.
- [docs/testing_and_ci.md](docs/testing_and_ci.md): local dev commands, docs-as-code checks, and GitHub Actions behavior.

## Quick Start

### Docker

The most reproducible path is Docker:

```bash
docker build -t power-stability-radius .
docker run --rm power-stability-radius
```

Run a small bundled compute example:

```bash
docker run --rm \
  -v "$PWD/run_artifacts:/app/run_artifacts" \
  power-stability-radius \
  python entry_points/power_stability_radius.py \
    --config conf/config.yaml \
    compute \
    --input data/input/ieee30.m \
    --slack-bus 0 \
    --base-dispatch case
```

For large PGLib or UnitCommitment.jl inputs, mount the data directory into
`/app/data/input` or pass absolute paths inside a mounted volume.

### Local Python

Install the project with Poetry:

```bash
poetry install
```

Run a compute pass:

```bash
poetry run python entry_points/power_stability_radius.py \
  --config conf/config.yaml \
  compute \
  --input data/input/pglib_opf_case30_ieee.m \
  --slack-bus 0 \
  --base-dispatch case
```

Run local checks:

```bash
poetry run python -m pytest -q
poetry run ruff format --check .
```

Artifacts are written under `run_artifacts/` by default. The exact subdirectory pattern depends on the entry point: the main CLI creates per-run folders such as `run_artifacts/compute/<timestamp>/`, while standalone scripts usually write into `run_artifacts/<module>/` unless an explicit output directory is requested.

## Environment And Reproducibility

The package targets Python 3.10 or newer and uses Poetry for dependency
resolution. `poetry.lock` pins the Python dependencies used for local runs and
CI. The solver-sensitive defaults are documented in
[docs/configuration.md](docs/configuration.md), including random seeds, HiGHS
settings, OPF headroom, and Monte Carlo sampling controls.

The project is distributed under the MIT License; see [LICENSE](LICENSE).
Academic citation metadata is provided in [CITATION.cff](CITATION.cff).
