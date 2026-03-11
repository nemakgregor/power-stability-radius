# Repository Structure

This document provides an annotated map of the repository layout, explaining the role of every important directory and file.

> Cross-references: [architecture.md](architecture.md) for component interactions, [developer_guide.md](developer_guide.md) for extension guidance.

---

## Top-Level Overview

```
power-stability-radius/
├── src/                           # Main source code
│   ├── power_stability_radius.py  # CLI entry point
│   └── stability_radius/          # Core Python package
├── experiments/                   # Experiment scripts and outputs
├── tests/                         # pytest test suite
├── conf/                          # Configuration files
├── data/                          # Input data (MATPOWER, UnitCommitment.jl)
├── docs/                          # This documentation
├── .github/                       # CI configuration
├── pyproject.toml                 # Poetry project configuration
├── README.md                      # Project README (Russian)
├── UNITS_CONTRACT.md              # Units contract documentation
└── AC Stability Radius TODO.md    # Development roadmap and TODO tracking
```

---

## `src/` — Source Code

### `src/power_stability_radius.py`
- **Category**: Entry point
- **Purpose**: Thin CLI wrapper that calls `cli.main()`
- **Dependencies**: `stability_radius.cli`
- **Lines**: ~41

### `src/stability_radius/` — Core Package

#### `__init__.py`
- **Purpose**: Package initialization, exports `compute_results_for_case` as the public API
- **Lines**: ~17

#### `cli.py`
- **Category**: Presentation layer
- **Purpose**: Full argparse CLI with 4 subcommands: `compute` (alias `demo`), `monte-carlo`, `report`, `table`
- **Key functions**: `main()`, `_handle_compute()`, `_handle_monte_carlo()`, `_handle_report()`, `_handle_table()`
- **Dependencies**: `config`, `workflows`, `verification`, `statistics`
- **Lines**: ~1200

#### `config.py`
- **Category**: Infrastructure
- **Purpose**: Configuration dataclasses and YAML loading with `extends` inheritance
- **Key classes**: `ProjectConfig`, `OPFConfig`, `ACFPFConfig`
- **Key function**: `load_config(path)` — loads YAML with `extends` resolution
- **Dependencies**: None (stand-alone)
- **Lines**: ~262

#### `workflows.py`
- **Category**: Core orchestration
- **Purpose**: Main computation pipeline orchestrating all phases (parse → base point → operators → radii)
- **Key function**: `compute_results_for_case()` — the central pipeline
- **Key classes**: `ACExtensionsConfig`, `DCExtensionsConfig`
- **Dependencies**: All other modules (parsers, base_point, dc, ac, radii)
- **Lines**: ~1400

#### `pp_helpers.py`
- **Category**: Support utility
- **Purpose**: Small canonical helpers for pandapower-like tables
- **Key functions**: `is_in_service()`, `bus_vn_kv()`, `resolve_slack_pos()`
- **Dependencies**: None (dependency-light by design)
- **Lines**: ~116

#### `metrics_analysis.py`
- **Category**: Experiment / analysis
- **Purpose**: Comparative evaluation pipeline — stability radii vs baseline metrics with Spearman correlation and precision-at-k
- **Key functions**: `build_unified_dataframe()`, `compute_rank_correlations()`, `compute_precision_at_k()`, `main()`
- **Dependencies**: `workflows`, `verification.monte_carlo`, `metrics.ac_baselines`
- **Lines**: ~568

---

### `src/stability_radius/ac/` — AC Model

#### `ac_model.py`
- **Category**: Core domain logic
- **Purpose**: AC power flow operator — Ybus construction, reduced PF Jacobian with PV/PQ bus handling, LU factorization, adjoint solves
- **Key class**: `ACOperator` — sparse Ybus, Jacobian, LU, adjoint solve methods
- **Key function**: `build_ac_operator()` — constructs ACOperator from pandapower network and AC PF base point
- **Internal functions**: `_build_ybus_pu()`, `_build_reduced_pf_jacobian_mw_per_unit()`, `_detect_pv_buses()`
- **Dependencies**: `dc.dc_model` (for trafo helpers), `pp_helpers`, `scipy.sparse`
- **Lines**: ~824

---

### `src/stability_radius/dc/` — DC Model

#### `dc_model.py`
- **Category**: Core domain logic
- **Purpose**: DC power flow operator — B-matrix assembly (lines + trafos + impedances), PTDF computation, phase-shifter support
- **Key class**: `DCOperator` — sparse B_red LU, incidence matrix W, sensitivity methods
- **Key functions**: `build_dc_operator()`, `build_dc_matrices()`, `trafo_tap_ratio()`, `trafo_x_total_ohm()`
- **Dependencies**: `pp_helpers`, `scipy.sparse`
- **Lines**: ~957

---

### `src/stability_radius/base_point/` — Base Point Computation

#### `types.py`
- **Purpose**: Frozen dataclasses `BasePointDC` and `BasePointAC` storing computed base operating points
- **Lines**: ~60

#### `dc.py`
- **Purpose**: DC base point computation from case dispatch or DC OPF
- **Key function**: `compute_dc_base_point()`
- **Dependencies**: `pypsa_opf`, `dc.dc_model`

#### `ac.py`
- **Purpose**: AC base point computation from AC PF
- **Key function**: `compute_ac_base_point()`
- **Dependencies**: `pypsa_pf`, `pandapower_tools`

#### `pypsa_opf.py`
- **Purpose**: DC OPF via PyPSA + HiGHS solver (converts pandapower → PyPSA network)
- **Key function**: `solve_dc_opf_base_flows_from_pandapower()`
- **Key class**: `PyPSAOPFResult`
- **Dependencies**: `pypsa`, `linopy`, pandapower
- **Lines**: ~564

#### `pypsa_pf.py`
- **Purpose**: AC PF base point via pandapower.runpp() with 3-attempt retry cascade
- **Key function**: `solve_ac_pf_base_point_from_pandapower()`
- **Key class**: `PyPSAAPFResult`
- **Dependencies**: pandapower
- **Lines**: ~1091

#### `pandapower_opp.py`
- **Purpose**: AC Feasibility Power Flow (FPF) via pandapower.runopp() with quadratic feasibility cost
- **Key function**: `solve_ac_fpf()`
- **Key class**: `ACFPFConfig`
- **Dependencies**: pandapower
- **Lines**: ~742

#### `pandapower_tools.py`
- **Purpose**: Shared pandapower utilities — lossless policy, slack bus resolution, generator dispatch application
- **Key functions**: `apply_lossless_policy_to_pandapower_net()`, `resolve_slack_bus_id()`, `ensure_ext_grid_at_slack()`, `apply_gen_dispatch_to_pandapower_net()`
- **Dependencies**: pandapower (import-time for `create_ext_grid`)
- **Lines**: ~350

---

### `src/stability_radius/radii/` — Radius Computation

#### `__init__.py`
- **Purpose**: Public API exports: `compute_l2_radius`, `compute_ac_l2_radius`, `compute_sigma_radius`, `compute_ac_sigma_radius`, `compute_ac_metric_radius`

#### `common.py`
- **Purpose**: Shared per-line data structures and limit estimation
- **Key class**: `LineBaseQuantities` — per-line base flows, limits, margins
- **Key function**: `estimate_line_limit_mva_with_flag()`, `get_line_base_quantities()`, `line_key()`

#### `core_l2.py`
- **Purpose**: Pure L2 certificate math (no pandapower dependency)
- **Key class**: `L2RadiusCertificate`
- **Key functions**: `compute_l2_certificate_from_H()`, `l2_norm_projected_ones_complement()`
- **Lines**: ~205

#### `l2.py`
- **Purpose**: DC L2 radius computation (wraps DCOperator + core_l2)
- **Key function**: `compute_l2_radius()`

#### `ac_l2.py`
- **Purpose**: AC L2 radius via ACOperator adjoint solves, chunked processing
- **Key function**: `compute_ac_l2_radius()`
- **Lines**: ~369

#### `probabilistic.py`
- **Purpose**: DC sigma-radius and Gaussian overload probability
- **Key functions**: `compute_sigma_radius()`, `flow_stddev()`, `overload_probability_symmetric_limit()`
- **Lines**: ~192

#### `ac_sigma_radius.py`
- **Purpose**: AC sigma-radius using precomputed h-vectors and per-bus sigma arrays
- **Key function**: `compute_ac_sigma_radius()`
- **Lines**: ~339

#### `metric.py`
- **Purpose**: DC metric radius with SPD weight matrix (Cholesky factorization)
- **Key function**: `compute_metric_radius()`
- **Lines**: ~134

#### `ac_metric_radius.py`
- **Purpose**: AC metric radius under arbitrary SPD weight matrix
- **Key function**: `compute_ac_metric_radius()`

#### `nminus1.py`
- **Purpose**: DC N-1 contingency radius with optional Woodbury sensitivity update
- **Key function**: `compute_nminus1_radius()`

#### `ac_feasibility.py`
- **Purpose**: AC feasibility check for base operating point (verifies |S0| ≤ S_limit)

---

### `src/stability_radius/verification/` — Verification

#### `types.py`
- **Purpose**: `VerificationResult` dataclass with status enums (BASE_OK, RADIUS_OK, SOUND_PASS, etc.)

#### `status.py`
- **Purpose**: High-level summary status mapping (OK, TRIVIAL_RADIUS, BASE_INFEASIBLE, CERT_UNSOUND, MC_INCONCLUSIVE)
- **Key function**: `summarize_status()`

#### `verify_certificate.py`
- **Purpose**: Deterministic certificate soundness check

#### `verify_worst_case.py`
- **Purpose**: Worst-case perturbation verification against full AC PF

#### `monte_carlo.py`
- **Purpose**: Monte Carlo verification engine for both DC and AC modes
- **Key function**: `run_monte_carlo_verification()`

#### `ac_monte_carlo_sigma.py`
- **Purpose**: AC Monte Carlo specifically for sigma-radius validation

#### `generate_report.py`
- **Purpose**: Multi-case Markdown report generator

---

### `src/stability_radius/metrics/` — Baseline Metrics

#### `ac_baselines.py`
- **Purpose**: Baseline robustness metrics — loading ratio, headroom (MVA), Cantelli upper bound
- **Key function**: `compute_baseline_metrics()`

---

### `src/stability_radius/parsers/` — Input Parsers

#### `matpower.py`
- **Purpose**: MATPOWER .m file → pandapower network conversion
- **Key function**: `load_network(path)` — uses the repository MATPOWER parser, then pandapower `from_ppc()`

#### `uc_jl.py`
- **Purpose**: UnitCommitment.jl JSON → per-bus sigma arrays from hourly demand data
- **Key function**: `load_sigma_from_uc_jl()`

---

### `src/stability_radius/statistics/` — Output Formatting

#### `table.py`
- **Purpose**: ASCII/CSV/Markdown table formatter for results.json

---

### `src/stability_radius/utils/` — Utilities

#### `download.py`
- **Purpose**: PGLib case file downloader from GitHub

---

## `experiments/` — Experiment Scripts

| File | Purpose | Paper Reference |
|------|---------|-----------------|
| `run_pglib_sweep.py` | DC vs AC radius comparison across PGLib cases | Fig. 1 |
| `run_sigma_radius.py` | Deep sigma-radius analysis with heterogeneous uncertainty | Fig. 2, Table 2 |
| `run_worst_case_verify.py` | Worst-case perturbation validation | Verification section |
| `run_scalability.py` | Wall-clock time vs network size | Scalability analysis |
| `collect_results.py` | Result collection and LaTeX table generation | Tables |
| `plot_radius_distribution.py` | Radius distribution visualization | Supplementary |
| `plot_sigma_vs_time.py` | Sigma vs time plots | Supplementary |
| `plot_worst_case_heatmap.py` | Worst-case heatmap visualization | Supplementary |
| `README.md` | Experiment documentation | — |

### `experiments/configs/`
Experiment-specific YAML configurations (case lists, sigma settings, data paths).

### `experiments/output/`
Generated experiment outputs (JSON results, plots, CSV tables, logs). Contains completed experiment runs in named subdirectories.

---

## `tests/` — Test Suite

~40+ test files using pytest. Key test files:

| File | Tests |
|------|-------|
| `conftest.py` | Shared fixtures (case5, case14, case30 networks) |
| `test_dc_model.py` | DC operator construction and PTDF correctness |
| `test_radii_l2.py` | L2 radius computation |
| `test_radii_metric.py` | Metric radius computation |
| `test_radii_nminus1.py` | N-1 contingency radius |
| `test_radii_probabilistic.py` | Sigma-radius and overload probability |
| `test_ac_sigma_radius.py` | AC sigma-radius |
| `test_ac_radius_smoke.py` | AC radius smoke tests on real cases |
| `test_certificate_concept.py` | Mathematical certificate concept validation |
| `test_h_vector_fd.py` | h-vector validation via finite differences |
| `test_ac_jacobian_vs_pandapower.py` | Jacobian validation against pandapower |
| `test_unit_consistency_end_to_end.py` | End-to-end units consistency |
| `test_verify_worst_case.py` | Worst-case verification tests |
| `test_config_extends.py` | Configuration extends mechanism |

---

## `conf/` — Configuration Files

| File | Purpose |
|------|---------|
| `config.yaml` | Main operational config (extends config_shared) |
| `config_shared.yaml` | Shared defaults for all configurations |
| `config_compute.yaml` | Compute-specific settings |
| `config_dc_extensions.yaml` | DC extensions (N-1, probabilistic) |
| `config_monte_carlo.yaml` | Monte Carlo verification settings |
| `config_report.yaml` | Report generation settings |
| `experiments/case30.yaml` | Case-specific experiment for IEEE 30-bus |
| `experiments/case118.yaml` | Case-specific experiment for IEEE 118-bus |

---

## `data/` — Input Data

### `data/input/`
MATPOWER .m files from the PGLib-OPF benchmark library. Contains 20+ test cases ranging from 5 to 10000 buses:
- Small: case5_pjm (5), case14_ieee (14), case24_ieee_rts (24), case30_ieee (30)
- Medium: case57_ieee (57), case73_ieee_rts (73), case118_ieee (118), case200_activ (200), case300_ieee (300)
- Large: case500_goc, case588_sdet, case1354_pegase, case1888_rte, case1951_rte
- Very large: case2000_goc, case2383wp_k, case2736sp_k, case2853_sdet, case2869_pegase
- Extreme: case6468_rte, case6515_rte, case9241_pegase, case10000_goc

### `data/uc_jl/`
UnitCommitment.jl format JSON files with hourly bus demand data (used for per-bus sigma computation).

---

## `.github/workflows/ci.yml`
GitHub Actions CI pipeline that runs the pytest test suite on push and pull requests.

---

## Root Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Poetry project configuration (dependencies, package metadata) |
| `poetry.lock` | Locked dependency versions |
| `.gitignore` | Git ignore rules (venv, pycache, outputs) |
| `README.md` | Project README in Russian |
| `UNITS_CONTRACT.md` | Detailed units contract for all quantities in the project |
| `AC Stability Radius TODO.md` | Development roadmap and feature tracking (~37K chars) |
