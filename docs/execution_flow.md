# Execution Flow

This document describes the step-by-step execution flow for each major entry point in the project.

> Cross-references: [architecture.md](architecture.md) for component roles, [configuration.md](configuration.md) for parameter details.

---

## 1. CLI Entry Point

**File**: `entry_points/power_stability_radius.py` â†’ `entry_points/power_stability_radius.py`

```
python entry_points/power_stability_radius.py --config conf/config.yaml <command> [options]
```

`entry_points/power_stability_radius.py` defines `main()`, which:

1. Creates an `argparse.ArgumentParser` with `--config` (global), `--log-level`
2. Adds subparsers for four commands: `compute`, `monte-carlo`, `report`, `table`
3. Parses arguments
4. Loads the YAML config via `config.load_config(args.config)` (with `extends` resolution)
5. Sets up Python logging (level, format)
6. Delegates to the appropriate handler function

---

## 2. `compute` Command (alias: `demo`)

**Handler**: `cli._handle_compute(args, cfg)`

This is the primary computation entry point. Full execution trace:

### Step 2.1 â€” Argument resolution

```
input_path    = args.input or cfg.input_path
slack_bus     = args.slack_bus or cfg.slack_bus  (supports -1 for auto-detect)
base_dispatch = args.base_dispatch or cfg.base_dispatch  (case|dc_opf|acpf|ac_fpf)
output_path   = args.output or "results.json"
```

### Step 2.2 â€” Build extension configs

```python
ac_ext = ACExtensionsConfig(
    sigma_p_mw_source=...,    # "uniform" or "uc_jl"
    sigma_q_mvar_source=...,
    sigma_p_mw_uniform=...,
    sigma_q_mvar_uniform=...,
    metric_enabled=...,
    save_h_vectors=...,
)
dc_ext = DCExtensionsConfig(...)
```

### Step 2.3 â€” Call `compute_results_for_case()`

This is the core pipeline (see [Section 3](#3-core-computation-pipeline) below).

### Step 2.4 â€” Save results

```python
# Strip non-serializable numpy arrays
# Write results.json
# Optionally write h-vectors as .npz
# Print summary table
```

---

## 3. Core Computation Pipeline

**Function**: `workflows.compute_results_for_case()`
**File**: `src/stability_radius/workflows.py` (~1400 lines)

This is the heart of the project. Full step-by-step flow:

### Phase 1 â€” Input Parsing

```
1. Resolve input_path (expand user, resolve absolute)
2. Load network via parsers.matpower.load_network(path)
   â†’ pandapower net object
3. Resolve slack bus:
   - Auto-detect from ext_grid if -1
   - Validate against ext_grid if specified
   - ensure_ext_grid_at_slack() creates ext_grid if missing
```

### Phase 2 â€” DC Base Point

If `compute_dc=True` or `base_dispatch="dc_opf"`:

```
4. Compute DC base point:
   a. If base_dispatch == "case":
      - Extract bus injections directly from net.res_bus
      - Run pandapower.runpp() for "case" base flows
   b. If base_dispatch == "dc_opf":
      - Convert pandapower â†’ PyPSA network (pypsa_opf.py)
      - Solve DC OPF with HiGHS (linopy model)
      - Apply headroom_factor to line limits
      - Extract generator dispatch, line flows, bus injections
5. Build DCOperator:
   - Assemble B matrix from lines + trafos + impedances
   - Check connectivity via union-find
   - LU-factorize B_red
6. Consistency check:
   - Reconstruct flows from OPF injections via DCOperator
   - Compare with OPF-reported flows (tolerance check)
```

### Phase 3 â€” DC Radii Computation

```
7. Compute DC L2 radius:
   - compute_l2_radius(net, H_full, ...)
   - Per-line: r_i = margin_i / ||g_i||_2
   - With optional balanced projection
8. Optional DC extensions:
   a. Probabilistic (sigma-radius):
      - compute_sigma_radius(net, H_full, Sigma)
      - Sigma = diag(inj_std_mwÂ²)
   b. N-1 contingency:
      - compute_nminus1_radius(net, op, ...)
      - For each line outage: update sensitivities, recompute radii
   c. Metric radius:
      - compute_metric_radius(net, H_full, M)
```

### Phase 4 â€” AC Base Point

If `compute_ac=True`:

```
9. Compute AC base point:
   a. If base_dispatch in {case, dc_opf}:
      - Apply DC OPF gen dispatch to pandapower net
      - Run AC PF via pandapower.runpp()
      - 3-attempt retry cascade:
        Attempt 1: init=dc, enforce_q_lims=True
        Attempt 2: relaxed voltage bounds, init=flat
        Attempt 3: further relaxed bounds, init=flat
   b. If base_dispatch == "acpf":
      - Run AC PF directly on case data
      - Correct slack loss (distribute losses to maintain P balance)
   c. If base_dispatch == "ac_fpf":
      - Run AC Feasibility Power Flow (pandapower.runopp)
      - Quadratic cost: min Î£ (p_g - p_g0)Â²
      - Post-OPP PF validation
10. Extract: V_mag_pu, V_ang_rad, line_p0_mw, line_q0_mvar, line_p1_mw, line_q1_mvar
```

### Phase 5 â€” AC Radii Computation

```
11. Build ACOperator:
    - Build Ybus (series-only, optionally lossless)
    - Detect PV/PQ buses
    - Build reduced PF Jacobian
    - LU-factorize Jacobian
12. Compute AC L2 radius:
    - For each line Ã— {from, to}: construct adjoint RHS, solve J^T a = b
    - Chunked processing (ac_chunk_size lines per batch)
    - Extract h-vectors if save_h_vectors=True
13. AC feasibility check:
    - Verify |S0| â‰¤ S_limit at each end of each line
14. Optional AC sigma radius:
    - Build sigma arrays (from uniform or UnitCommitment.jl)
    - compute_ac_sigma_radius(h_vectors, s_limit, s0, sigma_p, sigma_q)
15. Optional AC metric radius:
    - M_diag = 1/ÏƒÂ² (inverse variance)
    - compute_ac_metric_radius(h_vectors, s_limit, s0, M_diag)
```

### Phase 6 â€” Results Assembly

```
16. Merge per-line results from all radii computations
17. Build __meta__ dict with full provenance:
    - Input path, slack bus, base dispatch, all settings
    - Base point metadata (P_gen, V_mag, solver status)
    - Compute time
18. Optionally attach h-vectors (_h_vectors key)
19. Return results dict
```

---

## 4. `monte-carlo` Command

**Handler**: `cli._handle_monte_carlo(args, cfg)`

### Execution flow:

```
1. Load results.json (from previous compute run)
2. Load input network (same case file)
3. Resolve parameters: mode (dc|ac), n_samples, seed, sigma values
4. Call run_monte_carlo_verification():
   a. Load base point from results
   b. Generate random perturbations:
      - DC: Î”p ~ N(0, Î£) with balanced projection (sum=0)
      - AC: Î”P ~ N(0, Ïƒ_pÂ²), Î”Q ~ N(0, Ïƒ_qÂ²) with balanced projection
   c. For each sample:
      - DC mode: Î”f = H Â· Î”p, check |f0 + Î”f| > c
      - AC mode: Apply perturbation to net, run AC PF, check |S| > c
   d. Aggregate: violation count, per-line overload fractions
   e. Compare with analytic certificate:
      - Are violations observed within the certified ball? (soundness check)
      - Do empirical overload probabilities match predictions? (probabilistic check)
5. Save verification results JSON
6. Print summary
```

---

## 5. `report` Command

**Handler**: `cli._handle_report(args, cfg)`

### Execution flow:

```
1. Parse --cases argument: list of (case_path, slack_bus) pairs
2. For each case:
   a. Run compute_results_for_case()
   b. Run Monte Carlo verification
   c. Run certificate verification
   d. Summarize status
3. Generate Markdown report:
   - Per-case: input info, base point summary, top-5 binding lines table
   - Per-case: verification status (OK, TRIVIAL_RADIUS, etc.)
   - Summary table across all cases
4. Save report.md
```

---

## 6. `table` Command

**Handler**: `cli._handle_table(args, cfg)`

### Execution flow:

```
1. Load results.json
2. Parse format options: --fmt (ascii|csv|markdown)
3. Select columns based on available data (DC only, AC only, or both)
4. Format table via helpers.reporting.table module
5. Print to stdout or save to file
```

---

## 7. Experiment Entry Points

### `run_pglib_sweep.py`

```
1. Load pglib_sweep.yaml config (case list, compute settings)
2. For each PGLib case:
   a. Load/download MATPOWER file
   b. Run compute_results_for_case() with DC+AC
   c. Save per-case JSON
   d. Extract summary metrics
3. Save summary.json
4. Generate comparison plot (DC vs AC radius)
```

### `run_sigma_radius.py`

```
1. Load experiment config (case path, UnitCommitment.jl data path)
2. Parse UnitCommitment.jl JSON â†’ per-bus sigma arrays
3. Run compute_results_for_case() with AC + sigma + metric + h-vectors
4. Save: results.json, sigma_arrays.json, hvectors.npz
5. Run AC Monte Carlo verification
6. Run worst-case verification
7. Generate plots: critical lines, flow vs limit, violation scale, topology heatmap
8. Generate table2_sigma_radius.csv
```

### `run_scalability.py`

```
1. Load pglib_sweep.yaml config
2. For each case, repeat N times:
   a. Time the compute_results_for_case() call
   b. Record wall-clock time, n_bus, n_line
3. Save scalability results CSV
```

### `run_worst_case_verify.py`

```
1. Load config with case path
2. Compute radii (to get worst-case perturbations)
3. For each critical line:
   a. Apply worst-case perturbation to base dispatch
   b. Run AC PF on perturbed network
   c. Compare predicted vs actual |S|
4. Report verification results
```

---

## 8. Metrics Analysis Pipeline

**Entry point**: `python entry_points/metrics_analysis.py`

```
1. Parse CLI args (--input, --sigma-p, --sigma-q, --mc-samples, etc.)
2. Step 1: Compute all AC radii via compute_results_for_case()
3. Step 2: Run MC with track_per_line_overloads=True
4. Step 3: Compute baseline metrics (loading_ratio, headroom, Cantelli)
5. Step 4: Build unified DataFrame (one row per line, all metrics as columns)
6. Compute Spearman rank correlations (each metric vs empirical overload prob.)
7. Compute precision-at-k (top-k most dangerous lines by each metric)
8. Generate plots: scatter, bar chart, histograms
9. Save: results.json, mc_verification.json, unified_per_line_metrics.csv,
   spearman_correlations.csv, precision_at_k.csv, plots
```

---

## 9. Error Handling and Recovery

### AC PF Retry Cascade

The AC PF solver uses a 3-attempt cascade with progressively relaxed settings:

| Attempt | Init | V bounds | Q limits |
|---------|------|----------|----------|
| 1 (primary) | dc | [0.9, 1.1] | enforced |
| 2 (relaxed_v) | flat | [0.85, 1.15] | relaxed |
| 3 (relaxed_all) | flat | [0.80, 1.20] | relaxed |

### AC FPF Retry Cascade

Similar pattern for pandapower.runopp(), with voltage bounds relaxation and configurable timeout per attempt.

### Error propagation

- Parse errors â†’ immediate `ValueError`
- PF non-convergence â†’ retry cascade â†’ final `RuntimeError`
- Singular Jacobian â†’ `RuntimeError` with diagnostic message
- Disconnected network â†’ `ValueError` from union-find connectivity check
