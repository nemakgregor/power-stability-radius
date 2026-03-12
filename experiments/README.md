# Experiments

Scripts that produce tables, plots, and numerical results for the paper.

> **Not tests.** `tests/` contains pytest unit/integration tests that assert
> correctness (pass/fail). Experiments are measurement scripts — they do not
> test; they measure.

## Structure

```
experiments/
├── __init__.py
├── configs/                       # YAML configs for each experiment
│   ├── pglib_sweep.yaml
│   └── uc_jl_case118.yaml
├── run_pglib_sweep.py             # Exp 1: DC vs AC radius across PGLib
├── run_sigma_radius.py            # Exp 2: sigma-radius with UC.jl data
├── run_worst_case_verify.py       # Exp 3: worst-case verification
├── run_scalability.py             # Exp 4: wall-clock time vs network size
├── collect_results.py             # Aggregate JSON -> CSV summary table
├── plot_radius_distribution.py    # Box-plots of DC/AC radii per case
├── plot_sigma_vs_time.py          # Sigma-radius bar chart + timing
├── plot_worst_case_heatmap.py     # Heatmap of verification errors
├── output/                        # Generated results (git-ignored)
└── README.md                      # This file
```

## Running experiments

Each `run_*.py` script:
- Reads a config YAML from `experiments/configs/`.
- Calls library functions from `stability_radius.*`.
- Writes results to `run_artifacts/{experiment_name}/`.
- Does **not** import from `tests/`.

### Experiment 1 — PGLib sweep (DC vs AC radius)

```bash
python entry_points/run_pglib_sweep.py
python entry_points/run_pglib_sweep.py --config experiments/configs/pglib_sweep.yaml
```

Computes DC and AC L2 radii for each PGLib case in the config and writes
per-case JSON + `summary.json` to `run_artifacts/run_pglib_sweep/`.

### Experiment 2 — Sigma-radius with UC.jl data

```bash
python entry_points/run_sigma_radius.py
python entry_points/run_sigma_radius.py --config experiments/configs/uc_jl_case118.yaml
```

Downloads a UC.jl instance, extracts per-bus sigma, and computes AC
sigma-radius. Results go to `run_artifacts/run_sigma_radius/`.

### Experiment 3 — Worst-case verification

```bash
python entry_points/run_worst_case_verify.py --results run_artifacts/run_pglib_sweep/pglib_opf_case30_ieee.json
```

Requires a prior sweep run with `save_h_vectors=true`. Verifies the
worst-case perturbation via nonlinear AC PF for the top-K most critical
lines.

### Experiment 4 — Scalability

```bash
python entry_points/run_scalability.py --repeats 3
```

Measures DC and AC compute time for each PGLib case with repeated runs.
Results go to `run_artifacts/run_scalability/`.

## Collecting & plotting

```bash
# Aggregate all JSON results into a single CSV
python entry_points/collect_results.py

# Generate plots
python entry_points/plot_radius_distribution.py
python entry_points/plot_sigma_vs_time.py
python entry_points/plot_worst_case_heatmap.py
```

## Output directory

All outputs go to `run_artifacts/` which is git-ignored. Each
experiment creates its own subdirectory:

```
run_artifacts/
├── run_pglib_sweep/       # Exp 1 results + plots
├── run_sigma_radius/      # Exp 2 results + plots
├── run_worst_case_verify/ # Exp 3 results + plots
├── run_scalability/       # Exp 4 timing results
└── collect_results/       # Aggregated summary CSV
```

