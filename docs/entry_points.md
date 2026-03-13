# Entry Points Reference

This document describes every runnable module in `entry_points/`. For each entry point it records:

- what it does;
- which inputs it expects;
- which artifacts it produces;
- how to launch it.

If a new runnable script is added to `entry_points/`, this file must be updated in the same commit.

## General Rules

- The main supported interface of the repository is `entry_points/power_stability_radius.py`.
- `entry_points/power_stability_radius.py` is intentionally thin; the application-layer implementation lives in `src/stability_radius/application/cli.py`.
- The remaining scripts are dedicated experiment or analysis fronts.
- Reusable post-processing code now lives under `src/stability_radius/postprocess/`, not under `entry_points/`.
- Artifacts are written under `run_artifacts/` by default.

## Quick Map

| File | Purpose | Typical outputs |
|------|---------|-----------------|
| `entry_points/power_stability_radius.py` | Main CLI for `compute`, `monte-carlo`, `report`, and `table` | `results.json`, `monte_carlo_stats.json`, `verification_report.md`, tables |
| `entry_points/metrics_analysis.py` | Compare radii against baseline and practical metrics | CSV tables, JSON, PNG plots |
| `entry_points/n1_stability_demo.py` | Cost OPF vs Radius OPF vs SCOPF demonstration | CSV, TXT, PNG, `debug.log` |
| `entry_points/run_pglib_sweep.py` | Batch DC vs AC sweep over a case list | Per-case JSON, `summary.json`, `fig1_*` |
| `entry_points/run_sigma_radius.py` | AC sigma-radius experiment | `results.json`, `summary.json`, `sigma_arrays.json`, `hvectors.npz`, plots |
| `entry_points/run_worst_case_verify.py` | Nonlinear validation of worst-case perturbations | `*_worst_case.json`, `table3_summary.json`, plots |
| `entry_points/run_scalability.py` | Timing study versus network size | `scalability.json` |

## `entry_points/power_stability_radius.py`

**What It Does**

This is the main CLI of the project. It exposes:

- `compute` or `demo`: deterministic DC and/or AC radius computation for one case;
- `monte-carlo`: statistical verification for an already computed `results.json`;
- `report`: multi-case Markdown report generation from prepared results;
- `table`: formatting of an existing `results.json`.

This is the default entry point unless you specifically need one of the dedicated experiment pipelines.

**Inputs**

Shared inputs:

- `--config conf/config.yaml` or another YAML file with project defaults;
- optional global flags such as `--runs-dir`, `--run-dir-mode`, `--run-name`, `--allow-download`, OPF options, and logging options.

Per subcommand:

- `compute`: `--input`, `--slack-bus`, `--base-dispatch`, DC/AC flags, and export options;
- `monte-carlo`: `--results`, `--input`, `--mode`, sampling options, and sigma options;
- `report`: `report.cases` in YAML plus `--results-dir`;
- `table`: a path to `results.json`.

**Outputs**

Under `run_artifacts/<command>/<timestamp-or-run-name>/`:

- `compute`: `results.json`, `results_table.txt`, optional `results_table_dc.csv`, `results_table_ac.csv`, optional `h_vectors.npz`;
- `monte-carlo`: `monte_carlo_stats.json`;
- `report`: `verification_report.md`;
- `table`: `results_table.txt`.

Shared run artifacts:

- `debug.log`
- `argv.txt`
- `config_source.yaml`
- `config.json`
- `config.yaml`

**Example Invocation**

Basic `compute` run:

```bash
python entry_points/power_stability_radius.py \
  --config conf/config.yaml \
  compute \
  --input data/input/pglib_opf_case30_ieee.m \
  --slack-bus 0 \
  --base-dispatch case \
  --compute-dc 1 \
  --compute-ac 1
```

Monte Carlo verification:

```bash
python entry_points/power_stability_radius.py \
  --config conf/config.yaml \
  monte-carlo \
  --mode dc \
  --results run_artifacts/compute/latest/results.json \
  --input data/input/pglib_opf_case30_ieee.m \
  --slack-bus 0 \
  --n-samples 10000
```

Report generation:

```bash
python entry_points/power_stability_radius.py \
  --config conf/config.yaml \
  report \
  --results-dir verification/results \
  --out verification/report.md
```

Table formatting:

```bash
python entry_points/power_stability_radius.py \
  --config conf/config.yaml \
  table run_artifacts/compute/latest/results.json \
  --format sections \
  --radius-field radius_ac_l2
```

## `entry_points/metrics_analysis.py`

**What It Does**

Builds a comparative analysis between stability radii and baseline or practical metrics. The script:

- computes AC radii;
- runs Monte Carlo to estimate per-line empirical overload probability;
- builds a unified per-line table;
- computes rank correlations and precision-at-k;
- creates diagnostic plots.

**Inputs**

- required `--input` pointing to a MATPOWER or PGLib `.m` file;
- optional `--slack-bus`, `--base-dispatch`;
- sigma settings, either `--sigma-p` / `--sigma-q` or load-proportional `--sigma-p-scale`, `--sigma-q-scale`;
- `--mc-samples`, `--mc-seed`;
- `--top-k`;
- optional `--output-dir`.

**Outputs**

Under `run_artifacts/metrics_analysis/...`:

- `results.json`
- `mc_verification.json`
- `unified_per_line_metrics.csv`
- `spearman_correlations.csv`
- `precision_at_k.csv`
- `summary_metric_comparison.csv`
- optional `hidden_danger_*.csv`
- optional `worst_case_direction_verification.csv`
- PNG plots such as `spearman_bar.png`, `radius_histograms.png`, and `precision_at_k_curves.png`

**Example Invocation**

```bash
python entry_points/metrics_analysis.py \
  --input data/input/pglib_opf_case30_ieee.m \
  --slack-bus 0 \
  --sigma-p 1.0 \
  --sigma-q 1.0 \
  --mc-samples 10000 \
  --output-dir case30_metrics
```

## `entry_points/n1_stability_demo.py`

**What It Does**

Runs the dedicated three-regime demonstration comparing:

- Cost OPF
- Radius OPF
- screening-based SCOPF

It computes radii, worst-case behavior, and N-1 diagnostics for each regime and writes both tables and plots.

**Inputs**

- required `--input` pointing to a MATPOWER `.m` case;
- optional `--output-dir`;
- optional `--slack-bus`;
- scenario parameters: `--r-target`, `--n-iter`, `--scopf-iter`;
- probabilistic parameters: `--sigma-p`, `--sigma-q`;
- optional skip flags: `--skip-n1-screening`, `--skip-dc-n1`, `--skip-ac-n1-radius`;
- `--verbose` for DEBUG logging.

**Outputs**

Under `run_artifacts/n1_stability_demo/<name>/`:

- `comparison_summary.txt`
- `plot_radius_cdf.png`
- `plot_ac_n1_radius_cdf.png`
- `plot_n1_overloads.png`
- `plot_cost_security_tradeoff.png`
- per-regime CSV files such as `cost_opf_radii.csv`, `dc_n1_cost_opf.csv`, `ac_n1_radii_cost_opf.csv`, `n1_screening_cost_opf.csv`
- `debug.log`

**Example Invocation**

```bash
python entry_points/n1_stability_demo.py \
  --input data/input/pglib_opf_case118_ieee.m \
  --output-dir n1_demo_case118 \
  --n-iter 2 \
  --scopf-iter 2 \
  --sigma-p 5.0 \
  --sigma-q 2.0
```

## `entry_points/run_pglib_sweep.py`

**What It Does**

Experiment 1: runs a case list from configuration and compares DC and AC radii on a shared base point. Use it when you need a multi-network summary.

**Inputs**

- `--config`, typically `experiments/configs/pglib_sweep.yaml`;
- optional `--reuse-dir` to reuse already computed per-case JSON files.

The YAML normally defines:

- `cases`
- `compute`
- `data_dir`
- `artifacts_root`

**Outputs**

Under `run_artifacts/run_pglib_sweep/...`:

- per-case JSON such as `pglib_opf_case30_ieee.json`
- `summary.json`
- `fig1_dc_vs_ac_radius.png`
- `fig1_dc_vs_ac_radius.pdf`
- `debug.log`

**Example Invocation**

```bash
python entry_points/run_pglib_sweep.py \
  --config experiments/configs/pglib_sweep.yaml
```

With reuse of existing outputs:

```bash
python entry_points/run_pglib_sweep.py \
  --config experiments/configs/pglib_sweep.yaml \
  --reuse-dir run_artifacts/run_pglib_sweep
```

## `entry_points/run_sigma_radius.py`

**What It Does**

Experiment 2: computes the AC sigma-radius around an average operating point. The script:

- derives sigma from UC.jl profiles or a synthetic source;
- solves the base AC operating point;
- computes AC L2 and AC sigma radii;
- runs worst-case verification;
- runs tightened-limit Monte Carlo;
- builds tables and plots.

**Inputs**

- `--config` with an experiment YAML such as `experiments/configs/uc_jl_case118.yaml`;
- the YAML defines the case, data paths, sigma source, verification settings, and plotting settings.

**Outputs**

Under `run_artifacts/run_sigma_radius/...`:

- `results.json`
- `summary.json`
- `sigma_arrays.json`
- `hvectors.npz`
- `table2_sigma_radius.csv`
- `verification_results.json`
- `mc_tightened_limit.json`
- `validation.json`
- plots `fig_critical_lines.*`, `fig_flow_vs_limit.*`, `fig_violation_scale.*`, `topology_sigma_radius.*`

**Example Invocation**

```bash
python entry_points/run_sigma_radius.py \
  --config experiments/configs/uc_jl_case118.yaml
```

## `entry_points/run_worst_case_verify.py`

**What It Does**

Experiment 3: selects bottleneck lines from existing results and checks analytical worst-case perturbations with full nonlinear AC power flow at multiple scale factors.

**Inputs**

One of two input modes:

- `--sweep-dir` containing outputs from `run_pglib_sweep.py`;
- or `--results` with one or more per-case JSON files.

Additional options:

- optional `--output-dir`;
- optional `--scales`;
- optional `--top-k`;
- optional `--recompute` to recompute `h` vectors instead of loading an NPZ file;
- optional `--cases` to filter the processed case set.

**Outputs**

Under `run_artifacts/run_worst_case_verify/...`:

- `<case>_worst_case.json`
- `table3_summary.json`
- `validation_worst_case.json`
- `fig3_worst_case_verify.png`
- `fig3_worst_case_verify.pdf`

**Example Invocation**

```bash
python entry_points/run_worst_case_verify.py \
  --sweep-dir run_artifacts/run_pglib_sweep \
  --top-k 3 \
  --scales 0.8 0.9 1.0 1.1
```

Or for specific JSON files:

```bash
python entry_points/run_worst_case_verify.py \
  --results run_artifacts/run_pglib_sweep/pglib_opf_case30_ieee.json \
  --recompute
```

## `entry_points/run_scalability.py`

**What It Does**

Experiment 4: measures DC and AC wall-clock time on a set of cases with different network sizes.

**Inputs**

- `--config`, typically `experiments/configs/pglib_sweep.yaml`;
- `--repeats`, the number of repeated runs per case.

**Outputs**

Under `run_artifacts/run_scalability/...`:

- `scalability.json` with `n_bus`, `n_line`, and mean/std timing for DC and AC.

**Example Invocation**

```bash
python entry_points/run_scalability.py \
  --config experiments/configs/pglib_sweep.yaml \
  --repeats 3
```

## Non-Entry Helpers

The following modules are intentionally not listed as entry points anymore:

- `src/stability_radius/postprocess/table.py`
- `src/stability_radius/postprocess/collect_results.py`
- `src/stability_radius/postprocess/plot_radius_distribution.py`
- `src/stability_radius/postprocess/plot_sigma_vs_time.py`
- `src/stability_radius/postprocess/plot_worst_case_heatmap.py`

They are reusable library-side helpers. Import them from Python code, or invoke them explicitly as modules with `python -m stability_radius.postprocess.<module>` only when you need an ad hoc post-processing utility.

## Which Entry Point to Choose

- Use `entry_points/power_stability_radius.py` for standard single-case compute, verification, reporting, and table formatting.
- Use `entry_points/run_pglib_sweep.py`, `entry_points/run_sigma_radius.py`, `entry_points/run_worst_case_verify.py`, and `entry_points/run_scalability.py` for experiment-style or paper-style batch workflows.
- Use `entry_points/metrics_analysis.py` or `entry_points/n1_stability_demo.py` for dedicated research workflows with their own CLI surface.
