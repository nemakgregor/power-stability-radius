# Software Architecture

This document describes the software architecture of the **Power Stability Radius** project: how its modules are organized, how they interact, and what design decisions underpin the implementation.

For complementary perspectives see:
- [repository_structure.md](repository_structure.md) -- directory tree and file roles
- [execution_flow.md](execution_flow.md) -- end-to-end execution traces
- [mathematical_foundations.md](mathematical_foundations.md) -- formal problem definitions and formulas
- [algorithms_and_models.md](algorithms_and_models.md) -- step-by-step algorithmic descriptions
- [configuration.md](configuration.md) -- YAML config system and all parameter defaults

---

## 1. Main Components

The project is organized into twelve distinct subsystems, each encapsulated in its own Python module or package under `src/stability_radius/`.

### 1.1 Entry Point (`power_stability_radius.py`)

**File:** `entry_points/power_stability_radius.py`

A deliberately thin script that imports `stability_radius.cli.main` and delegates to it. This separation keeps the repository usable as a library (via `stability_radius.workflows.compute_results_for_case`) without importing CLI-only dependencies at import time.

```python
# Library API:
from stability_radius.workflows import compute_results_for_case

# CLI API:
from stability_radius.cli import main as cli_main
```

### 1.2 CLI Layer (`cli.py`)

**File:** `src/stability_radius/cli.py`

The command-line interface is built on `argparse` and provides four subcommands:

| Subcommand | Alias | Purpose |
|---|---|---|
| `compute` | `demo` | Run the full stability radius computation pipeline |
| `monte-carlo` | -- | Run Monte Carlo verification against previously computed results |
| `report` | -- | Generate a multi-case Markdown verification report |
| `table` | -- | Pretty-print or export an existing `results.json` as ASCII/CSV tables |

**Key design points:**

- A pre-parse step (`_preparse_config_path`) extracts the `--config` path before the main argument parse, allowing YAML defaults to be injected into the argparse default values.
- OmegaConf (`_cfg_get`) is used to read nested YAML keys with fallbacks to the programmatic defaults from `config.py`.
- If no subcommand is given on the command line, the CLI inspects `command:` in the YAML config to infer a default command.
- Each subcommand handler (`run_compute`, `run_monte_carlo`, `run_report`, `run_table`) writes run artifacts (config snapshots, argv) to a timestamped run directory before executing.

### 1.3 Configuration (`config.py`)

**File:** `src/stability_radius/config.py`

Centralizes all user-facing defaults and solver settings into frozen dataclasses:

| Dataclass | Responsibility |
|---|---|
| `LoggingConfig` | Run directory layout, console/file log levels |
| `HiGHSConfig` | HiGHS LP solver parameters (threads, seed, tolerances) |
| `OPFConfig` | DC OPF settings: HiGHS config, headroom factor, unconstrained line surrogate limit |
| `DCConfig` | DC model defaults: operator vs materialize mode, chunk size, dtype |
| `MonteCarloConfig` | Verification defaults: n_samples, seed, feasibility/cert tolerances |

**Module-level singletons** (`DEFAULT_LOGGING`, `DEFAULT_OPF`, `DEFAULT_DC`, `DEFAULT_MC`) serve as the canonical defaults for both programmatic and CLI usage.

**YAML loading with `extends` inheritance:**

The function `load_project_config(path)` implements a deterministic composition mechanism:

1. Load the YAML file at `path` via OmegaConf.
2. If the file contains an `extends:` key (string or list of paths), recursively load each base config.
3. Merge bases left-to-right, then overlay the current file's own keys.
4. Cycle detection prevents infinite `extends` loops.

This replaces Hydra's full composition engine with a minimal, deterministic alternative. See [configuration.md](configuration.md) for the full parameter reference.

### 1.4 Workflow Orchestrator (`workflows.py`)

**File:** `src/stability_radius/workflows.py` (~1410 lines)

The central function `compute_results_for_case()` orchestrates the entire computation pipeline. It accepts all parameters as explicit arguments (no global state) and returns a JSON-serializable `dict`.

**Pipeline stages:**

```
1. Resolve input file path (with optional download)
2. Parse MATPOWER .m file -> pandapower network
3. Validate thermal limit sources
4. Compute base operating point:
     case dispatch | DC OPF | AC PF | AC FPF
5. Build DC operator (B-matrix, PTDF)
6. Compute DC radii:
     L2 | sigma | metric | N-1 | probabilistic
7. Optionally build AC operator (Ybus, Jacobian)
8. Compute AC radii:
     AC L2 | AC sigma | AC metric
9. Merge per-line results + metadata
10. Return results dict
```

**Helper dataclasses defined in this module:**

- `DCExtensionsConfig` -- gates optional DC post-processing (sigma-radius, N-1 contingency radius)
- `ACExtensionsConfig` -- gates optional AC post-processing (sigma-radius, metric-radius, h-vector export)

**Internal helpers:**

- `_merge_line_results(*dicts)` -- merges multiple per-line result dictionaries with deterministic key ordering
- `_compute_projected_norms_from_operator()` -- chunked computation of balanced-subspace sensitivity norms
- `_extract_binding_end_data()` -- selects the binding (from/to) end for AC thermal limits
- `_build_sigma_arrays()` -- constructs per-bus sigma arrays from uniform or UnitCommitment.jl sources
- `_expand_h_reduced_to_full()` -- expands reduced-dimension h-vectors back to full bus ordering

### 1.5 Parsers (`parsers/`)

**Package:** `src/stability_radius/parsers/`

| Module | Responsibility |
|---|---|
| `matpower.py` | Parse MATPOWER `.m` files into pandapower networks. Uses the repository's deterministic MATPOWER parser and converts PPC -> pandapower via `from_ppc()` |
| `uc_jl.py` | Parse UnitCommitment.jl JSON instance files to extract per-bus sigma arrays from hourly demand data |

The parsers have no dependency on the radius computation modules and produce standard pandapower network objects or NumPy arrays. See [data_formats.md](data_formats.md) for input file schemas.

### 1.6 Base Point Computation (`base_point/`)

**Package:** `src/stability_radius/base_point/`

Computes the operating point around which the stability radius is certified.

| Module | Responsibility |
|---|---|
| `types.py` | Frozen dataclasses `BasePointDC` and `BasePointAC` -- immutable containers for DC/AC base regimes with `to_meta_dict()` for JSON serialization |
| `dc.py` | `compute_dc_base_point()` -- assembles a `BasePointDC` from either case dispatch data or DC OPF results |
| `ac.py` | `compute_ac_base_point()` -- assembles a `BasePointAC` from an AC power flow solution |
| `pypsa_opf.py` | DC OPF via PyPSA + HiGHS: converts pandapower network to PyPSA, solves LP, extracts flows/injections |
| `pypsa_pf.py` | AC PF via `pandapower.runpp()` with a 3-attempt retry cascade (flat init, DC init, relaxed tolerances) |
| `pandapower_opp.py` | AC Feasibility Power Flow via `pandapower.runopp()` (OPP) with quadratic feasibility cost functions |
| `pandapower_tools.py` | Shared utilities: lossless network policy enforcement, slack bus resolution, generator dispatch application |

**BasePointDC fields:** `bus_ids`, `bus_injections_mw`, `line_ids`, `line_flows_mw`, `line_limits_mw`, `gen_dispatch_mw_by_name`, `status`, `objective`.

**BasePointAC fields:** `bus_ids`, `vm_pu`, `va_rad`, `line_ids`, `p_from_mw`, `q_from_mvar`, `p_to_mw`, `q_to_mvar`, `s_limit_mva`, `pf_solver`, `pf_init`, `lossless`, `pf_attempt`, `pf_repairs`, `distributed_slack_requested`, `distributed_slack_used`.

### 1.7 DC Model (`dc/dc_model.py`)

**File:** `src/stability_radius/dc/dc_model.py`

Implements the `DCOperator` frozen dataclass, which encapsulates the lossless DC linear power flow model.

**B-matrix assembly:**

- Builds a sparse oriented incidence matrix `A` for all in-service lines, transformers, and impedances.
- Computes per-branch susceptance coefficients `b` (MW/rad) from reactance and nominal voltage.
- Handles transformer tap ratios (`tap_pos`, `tap_step_percent`) with MATPOWER-style DC approximation: `b_eff = b / tap`.
- Supports phase-shifting transformers via `shift_degree` (converted to radians).
- Forms `B = A^T diag(b) A` and eliminates the slack bus row/column to produce the reduced system.

**Key methods:**

| Method | Description |
|---|---|
| `flows_from_delta_injections(dp)` | Compute line flow changes from bus injection perturbations via sparse LU back-substitution |
| `materialize_H_full()` | Explicitly form the full PTDF matrix `H` (m x n_bus) -- required for N-1 radii |
| `row_norms_l2()` | Compute per-line L2 norms of PTDF rows without materializing `H` |
| `row_sensitivities_transposed(line_indices)` | Return transposed sensitivity block for a chunk of lines |

**Factory functions:**

- `build_dc_matrices(net, slack_bus)` -- returns raw sparse matrices
- `build_dc_operator(net, slack_bus)` -- returns a fully initialized `DCOperator` with LU factorization

### 1.8 AC Model (`ac/ac_model.py`)

**File:** `src/stability_radius/ac/ac_model.py`

Implements the `ACOperator` frozen dataclass, which encapsulates the linearized AC power flow model around a solved AC PF base point.

**Ybus and Jacobian construction:**

- Builds sparse admittance matrix `Ybus` from line/transformer impedances (with optional lossless policy: `r = 0`).
- Constructs the reduced AC power flow Jacobian of the form:

  ```
  J = [ dP/dtheta   dP/dV_pq ]
      [ dQ/dtheta   dQ/dV_pq ]
  ```

- PV buses (those controlled by generators or ext_grids) are excluded from the voltage magnitude variables and reactive power equations.
- The slack bus is excluded from all rows and columns.

**Key methods:**

| Method | Description |
|---|---|
| `solve_J(rhs)` | Forward solve: Jacobian * x = rhs (for sensitivity computation) |
| `solve_J_transpose(rhs)` | Adjoint solve: J^T * x = rhs (for efficient per-line sensitivity extraction) |

The adjoint solve is critical for the AC L2 radius computation, where per-line h-vectors are computed via `J^T`-solves rather than explicit Jacobian inversion. See Section 5.3 for details.

### 1.9 Radii Computation (`radii/`)

**Package:** `src/stability_radius/radii/`

The core mathematical engine. Each module implements one radius variant.

| Module | Function / Class | Description |
|---|---|---|
| `core_l2.py` | `l2_norm_projected_ones_complement()`, `L2RadiusResult` | Pure L2 certificate math with balanced subspace projection; no pandapower/operator dependencies. Testable in isolation |
| `l2.py` | `compute_l2_radius()` | DC L2 radius: wraps `DCOperator` + `core_l2` to produce per-line `r_i = (c_i - \|f0_i\|) / \|g_i\|_2` |
| `ac_l2.py` | `compute_ac_l2_radius()` | AC L2 radius: uses `ACOperator` adjoint solves, processes lines in chunks for memory efficiency. Returns per-line radii and h-vectors |
| `probabilistic.py` | `sigma_radius()`, `overload_probability_symmetric_limit()` | DC sigma-radius and Gaussian overload probability via the Q-function |
| `ac_sigma_radius.py` | `compute_ac_sigma_radius()` | AC sigma-radius with precomputed h-vectors and sigma-squared-weighted balanced projection |
| `metric.py` | `compute_metric_radius()` | DC metric radius with symmetric positive-definite (SPD) weight matrix via Cholesky factorization |
| `ac_metric_radius.py` | `compute_ac_metric_radius()` | AC metric radius using precomputed h-vectors and a weight matrix |
| `nminus1.py` | `compute_nminus1_l2_radius()` | N-1 contingency radius with optional sensitivity updates via Woodbury/LODF approximation |
| `ac_feasibility.py` | `check_ac_base_point_feasibility()` | AC feasibility check: validates whether the base operating point satisfies all thermal limits |
| `common.py` | `LineBaseQuantities`, `estimate_line_limit_mva()`, `line_key()` | Shared dataclass and utilities for per-line base quantities, thermal limit extraction, and result key formatting |

### 1.10 Verification (`verification/`)

**Package:** `src/stability_radius/verification/`

Provides independent validation of the computed certificates.

| Module | Responsibility |
|---|---|
| `monte_carlo.py` | Monte Carlo verification engine supporting both DC and AC modes. Draws random perturbations, evaluates flows, checks whether violations occur inside the certified ball |
| `ac_monte_carlo_sigma.py` | AC-specific Monte Carlo for sigma-radius validation with per-bus sigma-scaled perturbations |
| `verify_certificate.py` | Deterministic certificate soundness check: verifies that the analytic worst-case perturbation does not violate limits |
| `verify_worst_case.py` | Worst-case perturbation verification: computes the exact worst-case perturbation direction and checks the resulting flow |
| `types.py` | `VerificationResult` and component dataclasses (`BasePointCheck`, `RadiusCheck`, `SoundnessCheck`, `ProbabilisticCheck`, `OverallCheck`). Status constants: `OVERALL_OK`, `OVERALL_WARN`, `OVERALL_FAIL` |
| `status.py` | Summary status mapping logic and human-readable status descriptions |
| `generate_report.py` | Multi-case Markdown report generator: runs verification for each case and produces a structured document |

The verification subsystem is intentionally decoupled from the radius computation modules: it loads previously saved `results.json` files and re-parses the input network independently to validate correctness.

### 1.11 Metrics (`metrics/`)

**Package:** `src/stability_radius/metrics/`

| Module | Responsibility |
|---|---|
| `ac_baselines.py` | Baseline robustness metrics for comparison: loading ratio, headroom in MVA, Cantelli upper bound on overload probability |

These metrics serve as empirical baselines for validating the analytic certificates.

### 1.12 Statistics and Experiments

**Statistics** (`src/stability_radius/helpers/reporting/table.py`):
ASCII and CSV table formatter for `results.json` files. Supports both flat and sectioned (DC + AC) output formats with configurable column selection.

**Experiments** (`experiments/`):

| Script | Purpose |
|---|---|
| `run_pglib_sweep.py` | DC vs AC radius comparison across all PGLib-OPF benchmark cases |
| `run_sigma_radius.py` | Deep sigma-radius analysis with visualization |
| `run_scalability.py` | Wall-clock time scalability analysis across increasing network sizes |
| `run_worst_case_verify.py` | Worst-case perturbation verification experiments |
| `collect_results.py` | Result aggregation and LaTeX table generation for papers |
| `plot_radius_distribution.py` | Per-line radius distribution plots |
| `plot_sigma_vs_time.py` | Sigma-radius vs computation time visualization |
| `plot_worst_case_heatmap.py` | Worst-case verification heatmaps |

See [experiments_and_evaluation.md](experiments_and_evaluation.md) for experimental methodology and reproducibility procedures.

---

## 2. Interaction Between Components

### 2.1 Component Dependency Graph

The following diagram shows the primary import dependencies between components (arrows point from importer to importee):

```
power_stability_radius.py
    |
    v
  cli.py ------> workflows.py ------> parsers/matpower.py
    |                 |                 parsers/uc_jl.py
    |                 |
    |                 +------> base_point/
    |                 |            dc.py
    |                 |            ac.py
    |                 |            pypsa_opf.py
    |                 |            pypsa_pf.py
    |                 |            pandapower_opp.py
    |                 |            pandapower_tools.py
    |                 |            types.py
    |                 |
    |                 +------> dc/dc_model.py
    |                 |
    |                 +------> ac/ac_model.py
    |                 |
    |                 +------> radii/
    |                 |            core_l2.py
    |                 |            l2.py
    |                 |            ac_l2.py
    |                 |            probabilistic.py
    |                 |            ac_sigma_radius.py
    |                 |            metric.py
    |                 |            ac_metric_radius.py
    |                 |            nminus1.py
    |                 |            ac_feasibility.py
    |                 |            common.py
    |                 |
    |                 +------> metrics/ac_baselines.py
    |
    +------> config.py (used by all components)
    |
    +------> verification/
    |            monte_carlo.py
    |            generate_report.py
    |            verify_certificate.py
    |            verify_worst_case.py
    |            types.py, status.py
    |
    +------> statistics/table.py
```

### 2.2 Key Interaction Patterns

**CLI to Workflow:**
The CLI translates parsed `argparse.Namespace` values into explicit keyword arguments for `compute_results_for_case()`. There is no shared mutable state -- all configuration flows as function arguments.

**Workflow to Base Point:**
The workflow orchestrator selects a base-point builder based on the `base_dispatch` parameter:

| `base_dispatch` | DC base point builder | AC base point builder |
|---|---|---|
| `"case"` | `build_dc_base_point_case()` | `solve_ac_pf_base_point()` |
| `"dc_opf"` | `build_dc_base_point_dc_opf()` | `solve_ac_pf_base_point()` (reusing OPF dispatch) |
| `"acpf"` | `build_dc_base_point_from_acpf()` | `solve_ac_pf_base_point()` |
| `"ac_fpf"` | `build_dc_base_point_from_acpf()` | `solve_ac_fpf_base_point()` |

When both DC and AC are requested, the DC OPF dispatch is reused for the AC PF base point via `gen_dispatch_mw_by_name` to ensure consistency.

**Operators to Radii:**
The `DCOperator` and `ACOperator` dataclasses encapsulate the linear models. Radius modules receive these operators as arguments and call their methods (`solve_J_transpose`, `row_sensitivities_transposed`, `flows_from_delta_injections`) to compute sensitivities. This decouples the mathematical certificate computations from the network model construction.

**Verification Independence:**
Verification modules (`monte_carlo.py`, `generate_report.py`) are invoked independently of the `compute` pipeline. They load `results.json` and re-parse the `.m` file to validate results from scratch, ensuring the verification is not coupled to the computation code paths.

---

## 3. Execution / Control Flow

### 3.1 CLI Entry

```
python entry_points/power_stability_radius.py [--config conf/config.yaml] <command> [options]
```

**Control flow in `main()`:**

1. Pre-parse `--config` to obtain the YAML config path.
2. Load and resolve the YAML config via `load_project_config()` (with `extends` inheritance).
3. Build the argparse parser with YAML-derived defaults (`build_parser(cfg)`).
4. Parse the full argv. If no subcommand is provided, infer from `command:` in the YAML.
5. Optionally run pytest self-tests (`--run-tests 1`).
6. Dispatch to the appropriate handler: `run_compute()`, `run_monte_carlo()`, `run_report()`, or `run_table()`.

### 3.2 Compute Pipeline (`run_compute`)

1. **Setup:** Create timestamped run directory, configure logging, write config artifacts.
2. **Delegate:** Call `compute_results_for_case()` with all parameters from the CLI namespace.
3. **Post-process:** Extract non-JSON-serializable h-vectors (if present) and save as `.npz`.
4. **Output:** Write `results.json`, format and write ASCII/CSV tables, print radius summaries.
5. **Export:** Optionally copy `results.json` to a user-specified export path.

### 3.3 Workflow Pipeline (`compute_results_for_case`)

Detailed execution trace (see also [execution_flow.md](execution_flow.md)):

```
1.  Resolve input path (download if allowed and missing)
2.  load_network(input_path)                    [parsers/matpower.py]
3.  assert_line_limit_sources_present(net)      [radii/common.py]
4.  resolve_slack_bus_id(net, slack_bus)         [base_point/pandapower_tools.py]
5.  IF compute_dc:
      5a. Build DC base point:                  [base_point/dc.py or pypsa_opf.py]
      5b. build_dc_operator(net, slack_bus)      [dc/dc_model.py]
      5c. compute_l2_radius(dc_op, base_point)  [radii/l2.py -> core_l2.py]
      5d. IF probabilistic_enabled:
            sigma_radius(), overload_prob()      [radii/probabilistic.py]
      5e. IF nminus1_enabled:
            compute_nminus1_l2_radius()          [radii/nminus1.py]
6.  IF compute_ac:
      6a. Build AC base point:                  [base_point/ac.py + pypsa_pf.py]
      6b. check_ac_base_point_feasibility()     [radii/ac_feasibility.py]
      6c. build_ac_operator(net, slack_bus, ...) [ac/ac_model.py]
      6d. compute_ac_l2_radius(ac_op, ...)      [radii/ac_l2.py]
      6e. IF sigma arrays available:
            compute_ac_sigma_radius()            [radii/ac_sigma_radius.py]
      6f. IF metric_enabled:
            compute_ac_metric_radius()           [radii/ac_metric_radius.py]
7.  Merge DC + AC results into unified dict
8.  Attach __meta__ block (base points, timing, config)
9.  Return results dict
```

### 3.4 Monte Carlo Verification Flow

1. Load `results.json` and parse the input `.m` file independently.
2. Reconstruct the `DCOperator` or `ACOperator` from the network.
3. Draw `n_samples` random perturbations (Gaussian, optionally balanced).
4. For each sample, compute flows and check limit violations.
5. Aggregate statistics (violation rates, max violations).
6. Build `VerificationResult` with component-level statuses.
7. Write `monte_carlo_stats.json`.

### 3.5 Report Generation Flow

1. Parse `report.cases` from the YAML config (list of case specifications).
2. For each case, run verification (Monte Carlo + certificate check).
3. Aggregate results across cases.
4. Generate a structured Markdown report with per-case sections.

---

## 4. Data Flow Between Modules

### 4.1 Primary Data Flow Diagram

```
                    MATPOWER .m file
                         |
                         v
               parsers/matpower.py
              load_network(path)
                         |
                         v
                 pandapower network (net)
                    /          \
                   /            \
           DC path               AC path
              |                     |
              v                     v
     base_point/dc.py         base_point/ac.py
     build_dc_base_point_*    solve_ac_pf_base_point
              |                     |
              v                     v
        BasePointDC             BasePointAC
              |                     |
              v                     v
     dc/dc_model.py            ac/ac_model.py
     build_dc_operator()       build_ac_operator()
              |                     |
              v                     v
         DCOperator             ACOperator
              |                     |
     +--------+--------+    +------+--------+
     |        |        |    |      |        |
     v        v        v    v      v        v
  radii/   radii/   radii/ radii/ radii/  radii/
  l2.py   prob.py  nminus1 ac_l2  ac_sig  ac_met
     |        |        |    |      |        |
     v        v        v    v      v        v
     +--------+--------+----+------+--------+
                         |
                         v
              Per-line results dicts
                         |
                         v
             _merge_line_results()
                         |
                         v
               Unified results dict
              (JSON-serializable)
                    /         \
                   /           \
                  v             v
          results.json     verification/
          (persisted)      monte_carlo.py
                           verify_certificate.py
```

### 4.2 Data Representations at Each Stage

| Stage | Representation | Key Fields |
|---|---|---|
| Raw input | MATPOWER `.m` text file | Bus, branch, gen tables in MATLAB syntax |
| Parsed network | `pandapower.pandapowerNet` | `net.bus`, `net.line`, `net.trafo`, `net.gen`, `net.ext_grid` DataFrames |
| DC base point | `BasePointDC` (frozen dataclass) | `bus_injections_mw` (n,), `line_flows_mw` (m,), `line_limits_mw` (m,) |
| AC base point | `BasePointAC` (frozen dataclass) | `vm_pu` (n,), `va_rad` (n,), `p_from_mw` (m,), `q_from_mvar` (m,), `s_limit_mva` (m,) |
| DC operator | `DCOperator` (frozen dataclass) | Sparse B-matrix, LU factorization, incidence matrix A, susceptance vector b |
| AC operator | `ACOperator` (frozen dataclass) | Sparse Ybus, Jacobian J, LU factorization, PV/PQ bus masks |
| Per-line result | `dict[str, float]` | `radius_l2`, `flow0_mw`, `margin_mw`, `norm_g`, etc. |
| Merged results | `dict[str, Any]` | `line_0: {...}`, `line_1: {...}`, ..., `__meta__: {...}` |
| Verification | `VerificationResult` (frozen dataclass) | `base_point`, `radius`, `soundness`, `probabilistic`, `overall` checks |

### 4.3 Generator Dispatch Propagation

When `base_dispatch="dc_opf"` and both DC and AC are requested:

```
DC OPF (pypsa_opf.py)
    |
    +---> gen_dispatch_mw_by_name: [("gen_0", 50.0), ("gen_1", 120.0), ...]
    |
    +---> BasePointDC.gen_dispatch_mw_by_name
    |
    +---> pandapower_tools.apply_gen_dispatch(net, dispatch)
    |
    +---> pandapower.runpp(net)  [pypsa_pf.py]
    |
    +---> BasePointAC  (consistent with DC OPF dispatch)
```

This ensures that the AC operating point uses the same generator dispatch as the DC OPF, preventing mismatches between the two model paths.

---

## 5. Major Abstractions

### 5.1 Linear Operator Abstraction

Both `DCOperator` and `ACOperator` encapsulate a linear relationship between bus injection perturbations and line flow changes:

```
DC:  delta_f = H * delta_p      (PTDF-based, exact in DC model)
AC:  delta_f ~ J_f * J^{-1} * delta_s   (Jacobian-based, first-order)
```

The operators expose methods for:
- **Forward computation:** Given `delta_p`, compute `delta_f` (used by Monte Carlo verification).
- **Adjoint/row extraction:** Given line indices, extract sensitivity rows `g_i` (used by radius computation).

This abstraction allows the radius modules (`core_l2.py`, `l2.py`, `ac_l2.py`) to use a uniform mathematical interface regardless of whether the underlying model is DC or AC.

### 5.2 Per-Line Certificate Structure

All radius computations produce per-line results following a common structure:

```python
{
    "line_<idx>": {
        "flow0_mw": float,        # Base flow magnitude
        "limit_mw": float,        # Thermal limit
        "margin_mw": float,       # limit - |flow0|
        "norm_g": float,          # Sensitivity norm
        "radius_<variant>": float # Stability radius
    }
}
```

The global certificate radius is `r* = min_i radius_i` across all lines. The `line_key(idx)` function in `radii/common.py` provides the canonical key format.

### 5.3 Chunked Adjoint Computation

For AC radius computation, materializing the full sensitivity matrix would require O(m * n) memory. Instead, the project uses chunked adjoint solves:

```python
for chunk in line_chunks:
    # Build RHS for this chunk of lines
    rhs = flow_sensitivity_rhs(chunk)    # (n_reduced, chunk_size)
    # Solve J^T * h = rhs for h-vectors
    h_block = ac_op.solve_J_transpose(rhs)
    # Compute radius for this chunk
    radii[chunk] = margin[chunk] / norm(h_block)
```

This bounds peak memory to O(chunk_size * n) while still producing exact results for the linearized model. The chunk size is configurable via `--ac-chunk-size` (default: 256).

### 5.4 Balanced Subspace Projection

Power balance requires that injection perturbations satisfy `1^T * delta_p = 0`. The project handles this via a projection rather than an explicit constraint:

For a sensitivity row `g_i`, the effective dual norm on the balanced subspace is:

```
||Proj(g_i)||_2 = ||g_i - mean(g_i) * 1||_2
```

This is computed numerically as `sqrt(||g||^2 - (sum(g))^2 / n)`, which avoids forming the explicit projection matrix. The `core_l2.py` module exposes `l2_norm_projected_ones_complement()` for this purpose.

This projection makes the radius invariant to the choice of slack bus, since changing the slack bus adds a constant vector to all sensitivity rows, which is removed by the projection.

### 5.5 Frozen Dataclasses as Value Objects

Throughout the codebase, data containers are implemented as `@dataclass(frozen=True)`:

- `BasePointDC`, `BasePointAC`
- `DCOperator`, `ACOperator`
- `LineBaseQuantities`
- `VerificationResult` and all its component dataclasses
- All configuration dataclasses

Immutability ensures that computed results cannot be accidentally mutated after construction, and makes reasoning about data flow explicit. The `to_meta_dict()` / `to_dict()` methods provide controlled conversion to JSON-serializable dictionaries.

---

## 6. Separation of Concerns

### 6.1 Layered Architecture

The project follows a four-layer architecture:

```
+-----------------------------------------------------------+
|  Layer 4: Presentation                                    |
|    cli.py, statistics/table.py, experiments/*.py           |
+-----------------------------------------------------------+
|  Layer 3: Orchestration                                   |
|    workflows.py, verification/generate_report.py           |
+-----------------------------------------------------------+
|  Layer 2: Domain Logic                                    |
|    radii/*.py, verification/monte_carlo.py,                |
|    verification/verify_*.py, metrics/ac_baselines.py       |
+-----------------------------------------------------------+
|  Layer 1: Infrastructure / Models                         |
|    parsers/*.py, base_point/*.py,                          |
|    dc/dc_model.py, ac/ac_model.py,                         |
|    config.py, utils/                                       |
+-----------------------------------------------------------+
```

Each layer depends only on layers below it. In particular:
- Radius modules do not import CLI code.
- The DC/AC models do not import radius or verification code.
- Parsers have no dependency on operators, radii, or verification.

### 6.2 Mathematical vs Operational Code

A deliberate separation exists between pure mathematical code and operational/infrastructure code:

| Pure math (no I/O, no pandapower) | Operational (I/O, pandapower, logging) |
|---|---|
| `radii/core_l2.py` | `radii/l2.py`, `radii/ac_l2.py` |
| Balanced projection helpers | `base_point/pypsa_opf.py` |
| Radius formulas | `parsers/matpower.py` |

`core_l2.py` is explicitly designed to be testable in isolation with synthetic data, without requiring a pandapower network or any external dependencies beyond NumPy.

### 6.3 Computation vs Verification

The computation pipeline (`workflows.py` + `radii/`) and the verification pipeline (`verification/`) are intentionally separate:

- **Computation** produces `results.json` -- a certified radius for each line.
- **Verification** loads `results.json` and independently re-derives quantities to check correctness.

This separation means a bug in the computation code does not automatically propagate into the verification code, providing genuine independent validation.

### 6.4 DC vs AC Separation

DC and AC functionality is kept in separate modules:

- DC operator: `dc/dc_model.py`; AC operator: `ac/ac_model.py`
- DC radius: `radii/l2.py`; AC radius: `radii/ac_l2.py`
- DC sigma: `radii/probabilistic.py`; AC sigma: `radii/ac_sigma_radius.py`
- DC metric: `radii/metric.py`; AC metric: `radii/ac_metric_radius.py`

The shared mathematical core (`radii/core_l2.py`) is reused by both paths. The workflow orchestrator (`workflows.py`) manages the coordination between DC and AC computations, including dispatch propagation.

---

## 7. Key Architecture Decisions

### 7.1 Pandapower as Intermediate Representation

All network data is converted to pandapower's internal format (`pandapowerNet`) immediately after parsing. This provides:

- A standard, well-tested network representation with typed DataFrames (`net.bus`, `net.line`, `net.trafo`, `net.gen`).
- Access to pandapower's AC power flow solver (`runpp`), optimal power flow (`runopp`), and network manipulation utilities.
- Consistent bus/line indexing across all downstream modules.

The MATPOWER import path is deterministic: the repository parses the `.m` file
into a PPC structure and then calls pandapower's `from_ppc()` converter. There
is no optional runtime fallback chain here.

### 7.2 Sparse LU Factorization for O(n) Solves

Both the DC B-matrix and the AC Jacobian are factorized using `scipy.sparse.linalg.splu()` (SuperLU). Once the LU factorization is computed (one-time O(n^1.5) cost for typical power networks), each solve (forward or adjoint) costs O(n), making the per-line sensitivity extraction efficient even for networks with thousands of buses.

### 7.3 Chunked Processing for Memory Efficiency

For AC L2 radius computation, processing all `m` lines at once would require O(m * n) memory for the h-vector matrix. The project processes lines in configurable chunks (default 256), bounding peak memory to O(chunk_size * n). Similarly, DC sensitivity norms can be computed in chunked mode (`--dc-mode operator`) without materializing the full PTDF matrix.

### 7.4 Lossless Policy for Certificate Consistency

The AC certificate uses a series-only (lossless) network model (`r = 0`, no shunt elements) when `--ac-lossless 1` (default). This ensures consistency between the DC and AC certificate assumptions: both models see the same lossless network, differing only in the linearization point (flat voltage for DC, AC PF solution for AC). The `pandapower_tools.py` module enforces this policy by zeroing resistance and shunt elements before operator construction.

### 7.5 Balanced Disturbances via Projection

Rather than adding an explicit equality constraint `1^T * delta_p = 0` to the radius optimization (which would complicate the per-line closed-form solution), the project projects sensitivity vectors onto the balanced subspace. This is mathematically equivalent but computationally simpler: it preserves the per-line `r_i = margin_i / ||Proj(g_i)||_2` closed-form formula.

### 7.6 Binding-End Selection for AC Thermal Limits

AC thermal limits can be binding at either the from-end or the to-end of a line (since AC flows differ at the two ends due to losses). The project checks both ends and selects the binding end -- the one with less margin -- for the radius computation. This is implemented in `_extract_binding_end_data()` within `workflows.py`.

### 7.7 Deterministic Configuration Propagation

The project enforces a "determinism contract": certain parameters (for example
`opf.threads`, `unconstrained_line_nom_mw`, Monte Carlo `seed`) must be
identical whether the code is invoked programmatically (via `DEFAULT_*`
singletons) or via CLI/YAML. The `config.py` module documents these contracts
explicitly, and the CLI injects YAML-derived values into the same dataclass
constructors used by the library API.

### 7.8 Immutable Results Pipeline

The computation pipeline is structured as a pure function:

```
inputs (path, config) --> compute_results_for_case() --> dict (JSON-serializable)
```

No global state is mutated. All intermediate data (base points, operators) is local to the function scope. The returned dictionary is the sole output, which the CLI then persists to disk. This makes the computation deterministic and testable.

### 7.9 Verification as an Independent Subsystem

The verification modules are architecturally independent from the computation modules. They re-parse the input network, reconstruct operators, and re-derive base quantities from scratch. This independence is a deliberate design choice: if the computation pipeline has a bug that produces incorrect sensitivities, the verification pipeline will not inherit that bug (it constructs its own operator from the raw network data).

### 7.10 Extension via Configuration, Not Code Modification

New radius variants (sigma, metric, N-1) are controlled by flags (`--compute-dc-probabilistic`, `--ac-metric-enabled`, `--compute-nminus1`) rather than requiring code changes. The `DCExtensionsConfig` and `ACExtensionsConfig` dataclasses group these flags, and the workflow orchestrator conditionally executes the corresponding computation blocks. See [developer_guide.md](developer_guide.md) for guidance on adding new radius variants.
