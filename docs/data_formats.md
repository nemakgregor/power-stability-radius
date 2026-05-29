# Data Formats Reference

This document describes every data format consumed or produced by the
power-stability-radius project.  For each format the following are covered:

- **Purpose** -- why the format exists.
- **Schema / fields** -- with data types.
- **Example snippets** -- where helpful.
- **Parser / producer** -- exact module and function names.
- **Constraints and validation** -- invariants enforced by the code.

---

## Table of Contents

1. [Input Formats](#1-input-formats)
   1. [MATPOWER .m Files](#11-matpower-m-files)
   2. [UnitCommitment.jl JSON Files](#12-unitcommitmentjl-json-files)
   3. [YAML Configuration Files](#13-yaml-configuration-files)
2. [Output Formats](#2-output-formats)
   1. [Computation Results JSON (results.json)](#21-computation-results-json-resultsjson)
   2. [Monte Carlo Verification JSON](#22-monte-carlo-verification-json)
   3. [Verification Report (Markdown)](#23-verification-report-markdown)
   4. [h-Vectors (NumPy .npz)](#24-h-vectors-numpy-npz)
   5. [Nonlinear Validation Report](#25-nonlinear-validation-report)
   6. [ASCII Results Tables](#26-ascii-results-tables)
   7. [CSV Outputs](#27-csv-outputs)
   8. [Plot Outputs](#28-plot-outputs)
   9. [Run Artifacts](#29-run-artifacts)
   10. [Experiment Outputs](#210-experiment-outputs)
   11. [Debug Logs](#211-debug-logs)

---

## 1. Input Formats

### 1.1 MATPOWER .m Files

#### Purpose

Describe power system network topology, bus data, generators, branch
parameters, and generator cost curves.  These files originate from the
[PGLib-OPF](https://github.com/power-grid-lib/pglib-opf) benchmark library
(v23.07) and serve as the primary input for all stability-radius computations.

#### Location

```
data/input/
```

24 case files ranging from 5 to 10 000 buses:

| File | Buses |
|------|------:|
| `pglib_opf_case5_pjm.m` | 5 |
| `pglib_opf_case14_ieee.m` | 14 |
| `pglib_opf_case24_ieee_rts.m` | 24 |
| `pglib_opf_case30_ieee.m` | 30 |
| `pglib_opf_case57_ieee.m` | 57 |
| `pglib_opf_case73_ieee_rts.m` | 73 |
| `pglib_opf_case118_ieee.m` | 118 |
| `pglib_opf_case200_activ.m` | 200 |
| `pglib_opf_case300_ieee.m` | 300 |
| `pglib_opf_case500_goc.m` | 500 |
| `pglib_opf_case588_sdet.m` | 588 |
| `pglib_opf_case1354_pegase.m` | 1 354 |
| `pglib_opf_case1888_rte.m` | 1 888 |
| `pglib_opf_case1951_rte.m` | 1 951 |
| `pglib_opf_case2000_goc.m` | 2 000 |
| `pglib_opf_case2383wp_k.m` | 2 383 |
| `pglib_opf_case2736sp_k.m` | 2 736 |
| `pglib_opf_case2853_sdet.m` | 2 853 |
| `pglib_opf_case2869_pegase.m` | 2 869 |
| `pglib_opf_case6468_rte.m` | 6 468 |
| `pglib_opf_case6515_rte.m` | 6 515 |
| `pglib_opf_case9241_pegase.m` | 9 241 |
| `pglib_opf_case10000_goc.m` | 10 000 |
| `ieee30.m` | 30 |

#### Schema

Each `.m` file is valid MATLAB source defining a function that returns a
struct `mpc`.  The project's parser extracts the following fields:

**Scalars**

| Field | Type | Description |
|-------|------|-------------|
| `mpc.version` | string | MATPOWER version (typically `'2'`). |
| `mpc.baseMVA` | float | System-wide base apparent power (MVA). |

**`mpc.bus` matrix** -- one row per bus.

| Column | Index | Type | Description |
|--------|------:|------|-------------|
| `bus_i` | 0 | int | Bus number (1-indexed). |
| `type` | 1 | int | Bus type: 1 = PQ, 2 = PV, 3 = slack (reference). |
| `Pd` | 2 | float | Active power demand (MW). |
| `Qd` | 3 | float | Reactive power demand (MVAr). |
| `Gs` | 4 | float | Shunt conductance (MW demanded at V = 1.0 p.u.). |
| `Bs` | 5 | float | Shunt susceptance (MVAr injected at V = 1.0 p.u.). |
| `area` | 6 | int | Area number. |
| `Vm` | 7 | float | Voltage magnitude (p.u.). |
| `Va` | 8 | float | Voltage angle (degrees). |
| `baseKV` | 9 | float | Base voltage (kV). |
| `zone` | 10 | int | Loss zone. |
| `Vmax` | 11 | float | Maximum voltage magnitude (p.u.). |
| `Vmin` | 12 | float | Minimum voltage magnitude (p.u.). |

**`mpc.branch` matrix** -- one row per branch (line or transformer).

| Column | Index | Type | Description |
|--------|------:|------|-------------|
| `fbus` | 0 | int | "From" bus number. |
| `tbus` | 1 | int | "To" bus number. |
| `r` | 2 | float | Resistance (p.u.). |
| `x` | 3 | float | Reactance (p.u.). |
| `b` | 4 | float | Total line charging susceptance (p.u.). |
| `rateA` | 5 | float | MVA rating A (long term). **0 = unconstrained.** |
| `rateB` | 6 | float | MVA rating B (short term). |
| `rateC` | 7 | float | MVA rating C (emergency). |
| `ratio` | 8 | float | Transformer off-nominal turns ratio. **0 = line (not trafo).** |
| `angle` | 9 | float | Transformer phase shift angle (degrees). |
| `status` | 10 | int | 1 = in service, 0 = out of service. |
| `angmin` | 11 | float | Minimum angle difference (degrees). |
| `angmax` | 12 | float | Maximum angle difference (degrees). |

**`mpc.gen` matrix** -- one row per generator.

| Column | Index | Type | Description |
|--------|------:|------|-------------|
| `bus` | 0 | int | Bus number. |
| `Pg` | 1 | float | Active power output (MW). |
| `Qg` | 2 | float | Reactive power output (MVAr). |
| `Qmax` | 3 | float | Maximum reactive power (MVAr). |
| `Qmin` | 4 | float | Minimum reactive power (MVAr). |
| `Vg` | 5 | float | Voltage magnitude setpoint (p.u.). |
| `mBase` | 6 | float | Machine base (MVA). |
| `status` | 7 | int | 1 = in service, 0 = out of service. |
| `Pmax` | 8 | float | Maximum active power output (MW). |
| `Pmin` | 9 | float | Minimum active power output (MW). |

**`mpc.gencost` matrix** -- generator cost data.

| Column | Index | Type | Description |
|--------|------:|------|-------------|
| `model` | 0 | int | Cost model type (2 = polynomial). |
| `startup` | 1 | float | Startup cost ($). |
| `shutdown` | 2 | float | Shutdown cost ($). |
| `ncost` | 3 | int | Number of cost coefficients. |
| `c(n-1)...c0` | 4+ | float | Cost coefficients (highest to lowest order). |

#### Example Snippet

```matlab
function mpc = pglib_opf_case5_pjm
mpc.version = '2';
mpc.baseMVA = 100.0;

%% bus data
%  bus_i type  Pd     Qd     Gs  Bs area  Vm      Va    baseKV zone Vmax   Vmin
mpc.bus = [
  1  2   0.0    0.0   0.0 0.0 1 1.00000 0.00000 230.0 1 1.10000 0.90000;
  2  1 300.0   98.61  0.0 0.0 1 1.00000 0.00000 230.0 1 1.10000 0.90000;
  ...
];

%% branch data
%  fbus tbus   r       x       b      rateA rateB rateC ratio angle status
mpc.branch = [
  1  2 0.00281 0.0281 0.00712 400.0 400.0 400.0 0.0 0.0 1 -30.0 30.0;
  ...
];

%% generator data
mpc.gen = [
  1  20.0 0.0 30.0 -30.0 1.0 100.0 1 40.0 0.0;
  ...
];

%% generator cost data
mpc.gencost = [
  2 0.0 0.0 3  0.000000 14.000000 0.000000;
  ...
];
```

#### Parser

**Module:** `src/stability_radius/parsers/matpower.py`

**Key functions:**

| Function | Role |
|----------|------|
| `load_network(file_path, f_hz=50.0)` | Public entry point. Returns a pandapower network. |
| `_parse_matpower_m_file_to_ppc(path)` | Regex-based extraction of `version`, `baseMVA`, `bus`, `gen`, `branch` matrices into a PPC dict. |
| `_from_ppc_to_pandapower(ppc, f_hz)` | Converts the PPC dict to pandapower via `pandapower.converter.pypower.from_ppc.from_ppc()`. |
| `_attach_matpower_rateA_to_net_lines(ppc, net)` | Propagates MATPOWER `branch.rateA` (column 5) into `net.line['rateA']` for line-like branches (those with `tap == 0`). |

#### Constraints and Validation

- File extension must be `.m`.
- `mpc.bus` and `mpc.branch` matrices must be non-empty.
- All matrix rows must have consistent column counts.
- MATPOWER comments (`%` to end-of-line) are stripped before parsing.
- `baseMVA` must parse as a valid float.
- Branches with `ratio == 0` (or NaN) are treated as lines; others as transformers.
- `rateA == 0` is interpreted as **unconstrained** (not zero limit) -- a large finite surrogate (default `1e5` MW) is used.

---

### 1.2 UnitCommitment.jl JSON Files

#### Purpose

Provide hourly load profiles and generator capacity time series for deriving
per-bus injection standard deviation (sigma) arrays.  These sigma values are
used by the sigma-radius and probabilistic overload computations.

#### Location

```
data/uc_jl/case118.json
```

#### Schema

Top-level JSON object with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `SOURCE` | string | Citation / provenance text. |
| `Parameters` | object | Instance parameters (see below). |
| `Generators` | object | Generator definitions keyed by name (e.g., `"g1"`). |
| `Buses` | object | Bus definitions keyed by name (e.g., `"b1"`). |

**`Parameters` object:**

| Key | Type | Description |
|-----|------|-------------|
| `Version` | string | UC.jl format version (e.g., `"0.3"`). |
| `Power balance penalty ($/MW)` | float | Penalty cost for unserved load. |
| `Time horizon (h)` | int | Number of hourly timesteps. |

**`Buses.<name>` object:**

| Key | Type | Description |
|-----|------|-------------|
| `Load (MW)` | float or float[] | Active power demand. Scalar for constant load; array of length T for hourly profiles. |

**`Generators.<name>` object (selected fields):**

| Key | Type | Description |
|-----|------|-------------|
| `Bus` | string | Bus name this generator is connected to (e.g., `"b1"`). |
| `Max power (MW)` | float or float[] | Maximum generator output. Scalar or hourly array. |
| `Production cost curve (MW)` | float[] | Piecewise-linear cost curve MW breakpoints. |
| `Production cost curve ($)` | float[] | Corresponding cost values. |
| `Startup costs ($)` | float[] | Tiered startup costs. |
| `Startup delays (h)` | int[] | Delay thresholds for startup cost tiers. |
| `Ramp up limit (MW)` | float | Maximum ramp-up rate. |
| `Ramp down limit (MW)` | float | Maximum ramp-down rate. |
| `Initial status (h)` | int | Initial on/off status (negative = hours offline). |
| `Initial power (MW)` | float | Initial power output. |

#### Example Snippet

```json
{
  "Parameters": {
    "Version": "0.3",
    "Time horizon (h)": 36
  },
  "Buses": {
    "b1": { "Load (MW)": [45.3, 42.1, 39.8, ...] },
    "b2": { "Load (MW)": [12.0, 11.5, 11.0, ...] }
  },
  "Generators": {
    "g1": {
      "Bus": "b1",
      "Max power (MW)": 89.83,
      "Production cost curve (MW)": [7.73, 28.25, 48.78, 69.30, 89.83],
      "Production cost curve ($)": [784.5, 1378.1, 2003.4, 3214.5, 5077.9]
    }
  }
}
```

#### Parser

**Module:** `src/stability_radius/parsers/uc_jl.py`

**Key functions:**

| Function | Returns | Role |
|----------|---------|------|
| `load_sigma(file_path, bus_mapping=None, power_factor=0.9)` | `dict` with `sigma_p_mw`, `sigma_q_mvar`, `n_timesteps`, `bus_mapping`, `metadata` | Extracts per-bus sigma by computing `std(Load (MW))` across hours per bus and `std(Max power (MW))` per generator. Total `sigma_P = sqrt(sigma_load^2 + sigma_gen^2)`. `sigma_Q = sigma_P * tan(arccos(power_factor))`. |
| `load_hourly_profiles(file_path, bus_mapping=None, power_factor=0.9)` | `dict` with `load_p_mw` (n_bus, n_timesteps), `load_q_mvar`, `n_timesteps`, `n_bus`, `bus_mapping`, `metadata` | Returns full hourly load matrices. |

**Return schema for `load_sigma()`:**

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `sigma_p_mw` | `np.ndarray` | `(n_bus,)` | Per-bus active power standard deviation (MW). |
| `sigma_q_mvar` | `np.ndarray` | `(n_bus,)` | Per-bus reactive power standard deviation (MVAr). |
| `n_timesteps` | `int` | scalar | Maximum number of timesteps found. |
| `bus_mapping` | `dict[str, int]` | -- | Bus name to index mapping (natural sort). |
| `metadata` | `dict` | -- | Source path, bus/generator counts, power factor. |

#### Constraints and Validation

- File extension must be `.json`.
- Must contain a non-empty `Buses` section.
- Bus names are sorted numerically (natural sort) for deterministic ordering.
- `sigma_P` per bus is computed with `ddof=0` (population standard deviation).
- Multiple generators on the same bus accumulate variance: `sigma_gen[bus] = sqrt(sum(sigma_gi^2))`.
- Power factor must be in `(0, 1]`; default `0.9`.

---

### 1.3 YAML Configuration Files

#### Purpose

Control all runtime parameters through composable YAML configurations.
The project uses OmegaConf for loading and supports a custom `extends`
inheritance mechanism for configuration composition.

#### Location

Primary configurations:

```
conf/
  config.yaml              # Main entrypoint (composes all below)
  config_shared.yaml       # Shared: io, logging, opf, dc, ac, tolerances, table
  config_compute.yaml      # compute command defaults
  config_monte_carlo.yaml  # monte-carlo command defaults
  config_report.yaml       # report command defaults
  config_dc_extensions.yaml  # DC probabilistic and N-1 extensions

conf/experiments/
  case30.yaml              # Per-case experiment config
  case118.yaml             # Per-case experiment config

experiments/configs/
  pglib_sweep.yaml         # PGLib sweep experiment config
  uc_jl_case118.yaml       # UC.jl sigma-radius experiment config
  sigma_case2000_goc.yaml  # Sigma-radius for large cases
  sigma_case2736sp_k.yaml
  sigma_case2869_pegase.yaml
```

#### Composition Mechanism

The `extends` key specifies one or more base configuration files to merge:

```yaml
# config.yaml (main entrypoint)
extends:
  - ./config_shared.yaml
  - ./config_compute.yaml
  - ./config_monte_carlo.yaml
  - ./config_report.yaml
```

Resolution rules:
- Paths are resolved relative to the file containing the `extends` key.
- Multiple bases are merged left-to-right; later values override earlier ones.
- The local file's values override all base values.
- Cyclic references are detected and raise `ValueError`.

#### Schema: `config_shared.yaml`

```yaml
run_tests: true                    # bool - run pytest before commands

io:
  allow_download: false            # bool - allow downloading missing case files

logging:
  runs_dir: runs                   # str - base directory for run outputs
  level_console: INFO              # str - console log level
  level_file: DEBUG                # str - file log level
  run_dir_mode: timestamp          # str - "timestamp" | "overwrite"
  run_name: latest                 # str - used when run_dir_mode=overwrite

ac:
  pf_solver: pandapower            # str - "pandapower" | "pypsa"
  lossless: true                   # bool - enforce r=0 in PF and Jacobian
  basepoint_s_tol_mva: 1.0e-3     # float - MC base-point consistency tolerance

opf:
  solver_name: highs               # str - must be "highs"
  threads: 4                       # int - HiGHS threads
  random_seed: 42                  # int - HiGHS random seed
  headroom_factor: 0.98            # float - OPF line constraint security margin
  unconstrained_line_nom_mw: 1.0e5 # float - finite surrogate for unconstrained lines
  ext_grid_marginal_cost_base: 1000.0  # float - ext_grid cost in PyPSA DC OPF

dc:
  mode: operator                   # str - "operator" | "materialize"
  chunk_size: 256                  # int - vectorised batch size
  dtype: float64                   # str - "float64" | "float32"

tolerances:
  opf_dc_flow_consistency_tol_mw: 1   # float - OPF->DC flow tolerance (MW)
  opf_bus_balance_tol_mw: 1            # float - bus injection balance tolerance (MW)

table:
  format: sections                 # str - "sections" | "flat"
  columns: []                      # str[] - base columns for all sections
  dc_extra_columns:                # str[] - DC section columns
    - flow0_mw
    - p0_mw
    - p_limit_mw_est
    - margin_mw
    - norm_g
    - radius_l2
    - constraint_status_l2
    - certificate_radius_l2
  ac_extra_columns:                # str[] - AC section columns
    - ac_s_limit_mva
    - ac_s0_from_mva
    - ac_s0_to_mva
    - margin_ac_mva
    - "||h||2"
    - binding_end
    - radius_ac_l2
    - constraint_status_ac_l2
    - certificate_radius_ac_l2
```

#### Schema: `config_compute.yaml`

```yaml
compute:
  input: data/input/pglib_opf_case30_ieee.m    # str - input .m file path
  slack_bus: 0                                  # int - slack bus index
  base_dispatch: case                           # str - "case" | "dc_opf" | "acpf" | "ac_fpf"

  dc:
    compute: true                  # bool - compute DC radii
    inj_std_mw: 1.0                # float - Gaussian sigma for DC MC (MW)

  ac:
    compute: true                  # bool - compute AC radii
    chunk_size: 256                # int - AC computation batch size
    balance: true                  # bool - project onto sum(dp)=0 subspace
    pf_init: flat                  # str - "flat" | "dc" | "pp"
    sigma:
      sigma_p_mw_source: ""       # str - "" | "uniform" | "uc_jl"
      sigma_q_mvar_source: ""     # str - "" | "uniform" | "uc_jl"
      sigma_p_mw_uniform: 1.0     # float - uniform sigma_P (MW)
      sigma_q_mvar_uniform: 1.0   # float - uniform sigma_Q (MVAr)
    metric:
      enabled: false               # bool - compute metric-radius cross-check
    save_h_vectors: false          # bool - save h-vectors to .npz
    validation:
      nonlinear:
        enabled: false             # bool - replay top-k worst-case directions
        top_k: 20                  # int - number of smallest AC L2 radii
        scale_max: 5.0             # float - max scale searched
        tol: 0.01                  # float - scale binary-search tolerance
        max_iter: 20               # int - binary-search iterations

  output:
    export_results: ""             # str - copy results.json to this path
    save_csv: true                 # bool - save CSV alongside ASCII table
    max_rows: null                 # int|null - limit table rows
    table_columns: ""              # str - comma-separated column override
```

#### Schema: `config_monte_carlo.yaml`

```yaml
monte_carlo:
  mode: dc                         # str - "dc" | "ac"
  results: ""                      # str - path to results.json
  input: ""                        # str - path to input .m file
  slack_bus: 0                     # int

  sampling:
    n_samples: 50000               # int
    seed: 42                       # int
    chunk_size: 256                # int

  tolerances:
    feas_tol: 0.0                  # float - feasibility tolerance (MW/MVA)
    cert_tol: 1                    # float - certificate tolerance (MW/MVA)
    cert_max_samples: 5000         # int - max samples for soundness check

  dc:
    sigma_override_mw: null        # float|null - override sigma (MW)

  ac:
    sigma_p_mw: 1.0                # float - AC Gaussian sigma_P (MW)
    sigma_q_mvar: 1.0              # float - AC Gaussian sigma_Q (MVAr)
```

#### Schema: `config_report.yaml`

```yaml
report:
  strict: false                    # bool - fail on missing DC/AC fields
  io:
    results_dir: verification/results  # str
    out: verification/report.md        # str

  sampling:
    n_samples: 50000               # int
    seed: 42                       # int
    chunk_size: 256                # int

  tolerances:
    feas_tol: 0.0                  # float
    cert_tol: 1                    # float
    cert_max_samples: 5000         # int

  dc:
    sigma_override_mw: null        # float|null
  ac:
    sigma_p_mw: 1.0                # float
    sigma_q_mvar: 1.0              # float

  generate_plots: false            # bool

  cases:                           # list - cases to verify
    - id: case30                   # str - unique identifier
      input: data/input/pglib_opf_case30_ieee.m  # str
      results: case30.json         # str - relative to results_dir
      known_critical_pairs:        # list of [int, int] bus pairs
        - [1, 2]
        - [2, 4]
```

#### Parser

**Module:** `src/stability_radius/config.py`

| Function | Role |
|----------|------|
| `load_project_config(path, allow_missing=True)` | Public entry point. Loads YAML with `extends` resolution. |
| `_load_with_extends(path, stack)` | Recursive loader with cycle detection. |
| `_resolve_path(p, base_dir)` | Resolve relative paths against the config file's directory. |

#### Constraints

- Root must be a YAML mapping (not a sequence or scalar).
- The `extends` key must be a string or list of strings.
- Referenced base files must exist.
- Cyclic references are detected and raise `ValueError`.

---

## 2. Output Formats

### 2.1 Computation Results JSON (`results.json`)

#### Purpose

Primary output of the `compute` command and the `compute_results_for_case()`
workflow function.  Contains all computed per-line DC and AC stability radii,
base-point metadata, and configuration for reproducibility.

#### Producer

**Module:** `src/stability_radius/workflows.py`

**Function:** `compute_results_for_case()`

**CLI command:** `power_stability_radius compute`

Output location: `<runs_dir>/<run_id>/results.json`

#### Schema

Top-level JSON object with two kinds of keys:

1. **`__meta__`** -- configuration, metadata, and diagnostics.
2. **`line_<N>`** -- per-line results (one key per monitored line, where N is the pandapower line index).

##### `__meta__` Object

```json
{
  "__meta__": {
    "schema_version": 3,
    "input_path": "/absolute/path/to/pglib_opf_case30_ieee.m",
    "slack_bus": 0,
    "base_dispatch": "ac_fpf",
    "base_dispatch_requested": "ac_fpf",
    "allow_download": false,
    "compute_dc": true,
    "compute_ac": true,
    "dc": { ... },
    "ac": { ... },
    "base_point_dc": { ... },
    "base_point_ac": { ... },
    "opf": { ... },
    "acpf_slack_loss_correction_mw": 0.0,
    "ac_fpf_pg0_source": "case",
    "compute_time_sec": 12.345,
    "opf_bus_balance_abs_mw": 0.0,
    "opf_dc_flow_max_abs_diff_mw": 0.0,
    "opf_dc_flow_tol_mw": 1.0,
    "opf_bus_balance_tol_mw": 1.0,
    "opf_dc_consistency_passed": true
  }
}
```

**`__meta__.dc` object:**

| Key | Type | Description |
|-----|------|-------------|
| `mode` | string | `"operator"` or `"materialize"`. |
| `dtype` | string | NumPy dtype used (e.g., `"float64"`). |
| `chunk_size` | int | Vectorised batch size. |
| `inj_std_mw` | float | DC Gaussian injection standard deviation (MW). |
| `probabilistic_enabled` | bool | Whether DC probabilistic was requested. |
| `probabilistic_computed` | bool | Whether DC probabilistic was actually computed. |
| `nminus1_enabled` | bool | Whether N-1 was requested. |
| `nminus1_computed` | bool | Whether N-1 was actually computed. |
| `nminus1_update_sensitivities` | bool | LODF approximation flag. |
| `nminus1_islanding` | string | `"skip"` or `"raise"`. |

**`__meta__.ac` object:**

| Key | Type | Description |
|-----|------|-------------|
| `pf_solver` | string | `"pandapower"` or `"pypsa"`. |
| `pf_init` | string | `"flat"`, `"dc"`, or `"pp"`. |
| `lossless` | bool | Whether series resistance was zeroed. |
| `distributed_slack_requested` | bool | Whether distributed slack was requested in config/CLI. |
| `distributed_slack` | bool | Whether distributed slack was actually used by the successful solve. |
| `trafo_model` | string | Transformer model type (e.g., `"pi"`). |
| `chunk_size` | int | AC computation batch size. |
| `balance` | bool | Whether `sum(dp)=0` projection was applied. |
| `pf_status` | string | Power flow convergence status. |
| `pf_attempt` | string | Which PF attempt succeeded. |
| `pf_repairs` | string[] | List of PF repair actions applied (for example `distributed_slack_auto_disabled_large_network`). |
| `feasibility` | object\|null | AC base point feasibility check result. |
| `sigma_source` | string\|null | `"uniform"`, `"uc_jl"`, or null. |
| `sigma_p_mw` | float\|float[]\|null | Per-bus sigma_P (scalar if uniform, array if uc_jl). |
| `sigma_q_mvar` | float\|float[]\|null | Per-bus sigma_Q. |
| `sigma_n_timesteps` | int\|null | Number of UC.jl timesteps used. |
| `sigma_computed` | bool | Whether sigma-radius was computed. |
| `metric_enabled` | bool | Whether metric-radius was requested. |
| `metric_computed` | bool | Whether metric-radius was computed. |
| `save_h_vectors` | bool | Whether h-vectors were saved to .npz. |
| `nonlinear_validation_enabled` | bool | Whether nonlinear top-k replay was requested. |
| `nonlinear_validation_computed` | bool | Whether nonlinear top-k replay was run. |
| `nonlinear_validation_top_k` | int | Requested number of AC L2 lines to replay. |
| `nonlinear_validation_scale_max` | float | Maximum replay scale searched. |
| `nonlinear_validation_tol` | float | Scale tolerance for replay binary search. |
| `nonlinear_validation_max_iter` | int | Maximum binary-search iterations. |

**`__meta__.opf` object:**

| Key | Type | Description |
|-----|------|-------------|
| `solver` | string | `"highs"` or `"n/a"`. |
| `threads` | int | Number of solver threads. |
| `random_seed` | int | Solver random seed. |
| `headroom_factor_configured` | float | User-configured headroom factor. |
| `headroom_factor_used` | float | Actual headroom factor used (may differ after adaptive relaxation). |
| `unconstrained_line_nom_mw` | float | Finite surrogate for unconstrained lines. |
| `ext_grid_absorption_mw` | float | Slack bus absorption (MW). |

**`__meta__.base_point_dc`** and **`__meta__.base_point_ac`** objects contain
solver-specific metadata for reproducing the exact base operating point
(bus voltages Vm/Va, generator dispatch, etc.).

##### Per-Line `line_<N>` Objects

Each `line_<N>` key maps to a dict containing DC fields, AC fields, or both:

**DC fields** (present when `compute_dc=true`):

| Key | Type | Unit | Description |
|-----|------|------|-------------|
| `flow0_mw` | float | MW | Signed base-point line flow (bus0 -> bus1). |
| `p0_mw` | float | MW | Absolute base-point flow `|flow0_mw|`. |
| `p_limit_mw_est` | float | MW | Estimated thermal limit (MVA used as MW under DC lossless). |
| `is_unconstrained` | bool | -- | True if the line has no real thermal constraint. |
| `margin_mw` | float | MW | `p_limit_mw_est - p0_mw`. Can be negative. |
| `norm_g` | float | -- | L2 norm of the projected sensitivity row vector `||g||_2`. |
| `radius_l2` | float | MW | Signed DC L2 diagnostic distance: `margin_mw / norm_g`. |
| `constraint_status_l2` | string | -- | Constraint-level status such as `ok_finite`, `base_infeasible`, or `unconstrained_limit`. |
| `certificate_radius_l2` | float | MW | Nonnegative DC L2 certificate radius. |
| `signed_distance_l2` | float | MW | Signed diagnostic distance `margin_mw / norm_g`. |

**DC probabilistic fields** (present when `dc.probabilistic_enabled=true`):

| Key | Type | Unit | Description |
|-----|------|------|-------------|
| `sigma_flow` | float | MW | Standard deviation of line flow: `inj_std_mw * norm_g`. |
| `radius_sigma` | float | -- | Signed sigma diagnostic distance: `margin_mw / sigma_flow`. |
| `constraint_status_sigma` | string | -- | Constraint-level status for the sigma model. |
| `certificate_radius_sigma` | float | -- | Nonnegative sigma certificate radius. |
| `signed_distance_sigma` | float | -- | Signed diagnostic sigma distance. |
| `overload_probability` | float | [0,1] | Two-sided Gaussian overload probability. |

**DC N-1 fields** (present when `dc.nminus1_enabled=true`):

| Key | Type | Unit | Description |
|-----|------|------|-------------|
| `radius_nminus1` | float | MW | Signed effective N-1 diagnostic distance. Negative values indicate a post-contingency base infeasibility. |
| `constraint_status_nminus1` | string | -- | `ok_finite`, `ok_infinite`, or `post_contingency_infeasible`. |
| `certificate_radius_nminus1` | float | MW | Nonnegative N-1 certificate radius. |
| `signed_distance_nminus1` | float | MW | Signed N-1 diagnostic distance. |

**AC fields** (present when `compute_ac=true`):

| Key | Type | Unit | Description |
|-----|------|------|-------------|
| `ac_s_limit_mva` | float | MVA | Apparent power thermal limit. |
| `ac_s0_from_mva` | float | MVA | Base apparent power at "from" end. |
| `ac_s0_to_mva` | float | MVA | Base apparent power at "to" end. |
| `binding_end` | string | -- | `"from"` or `"to"` -- which end is closer to the limit. |
| `margin_ac_mva` | float | MVA | `ac_s_limit_mva - max(ac_s0_from_mva, ac_s0_to_mva)`. |
| `\|\|h\|\|2` | float | -- | L2 norm of the binding-end h-vector. |
| `radius_ac_l2` | float | MW | Signed AC L2 diagnostic distance: `margin_ac_mva / ||h||_2`. |
| `radius_ac_l2_linear` | float | MW | Linear first-order AC L2 diagnostic distance. |
| `constraint_status_ac_l2` | string | -- | Constraint-level AC L2 status. |
| `certificate_radius_ac_l2` | float | MW | Nonnegative linear AC L2 certificate radius. |
| `signed_distance_ac_l2` | float | MW | Signed diagnostic distance `margin_ac_mva / ||h||_2`. |
| `nondifferentiable_apparent_power` | bool | -- | True when the binding AC apparent-power base point has `|S0|` near zero; the signed radius is diagnostic, not a strict first-order certificate. |
| `radius_ac_l2_validated` | float | MW | Nonnegative radius retained after nonlinear replay; present when top-k replay is enabled. |
| `validation_scale_safe` | float | -- | Largest replay scale observed converged and non-violating. |
| `validation_scale_violation` | float | -- | Estimated first nonlinear violation scale relative to the linear boundary. |
| `nonlinear_conservatism_ratio` | float | -- | Same scale ratio; `>1` conservative, `<1` optimistic, `inf` no violation up to `scale_max`. |
| `pf_replay_status` | string | -- | Nonlinear replay PF status summary, e.g. `"converged"` or `"pf_failed"`. |
| `max_replay_rel_error` | float | -- | Maximum relative error between replayed nonlinear apparent power and linear prediction over converged replay points. |
| `nonlinear_validation_n_pf_calls` | int | -- | Number of PF calls used for this line's violation-scale search. |
| `linearization_status` | string | -- | Nonlinear validation or active-set status; `nonlinear_unvalidated`, `validated_local`, `nonlinear_optimistic`, or `invalid_active_set_changed_q_limit`. |
| `q_limit_hit` | bool | -- | True if the AC base PF detected a generator/ext_grid reactive-limit event. |
| `pv_pq_switch_detected` | bool | -- | True when Q-limit diagnostics indicate that the fixed PV/PQ active set is no longer a strict certificate model. |

**AC sigma fields** (present when sigma arrays are provided):

| Key | Type | Unit | Description |
|-----|------|------|-------------|
| `sigma_flow_mva` | float | MVA | Standard deviation of linearised flow: `||diag(sigma) * h||_2`. |
| `radius_ac_sigma` | float | -- | Signed AC sigma diagnostic distance: `margin_ac_mva / sigma_flow_mva`. |
| `constraint_status_ac_sigma` | string | -- | Constraint-level AC sigma status. |
| `certificate_radius_ac_sigma` | float | -- | Nonnegative AC sigma certificate radius. |
| `signed_distance_ac_sigma` | float | -- | Signed diagnostic AC sigma distance. |
| `overload_probability_ac` | float | [0,1] | One-sided Gaussian apparent-power overload probability. |

**AC metric fields** (present when `ac.metric.enabled=true`):

| Key | Type | Unit | Description |
|-----|------|------|-------------|
| `radius_ac_metric` | float | -- | Metric-radius with `M = diag(1/sigma^2)`. |
| `constraint_status_ac_metric` | string | -- | Constraint-level AC metric status. |
| `certificate_radius_ac_metric` | float | -- | Nonnegative AC metric certificate radius. |
| `signed_distance_ac_metric` | float | -- | Signed diagnostic AC metric distance. |

**AC worst-case fields:**

| Key | Type | Unit | Description |
|-----|------|------|-------------|
| `worst_case_dp_mw` | float[] | MW | Per-bus active power perturbation that achieves the radius. |
| `worst_case_dq_mvar` | float[] | MVAr | Per-bus reactive power perturbation. |

#### Constraints

- `line_<N>` keys are sorted numerically.
- `radius_l2` is `inf` when `norm_g` is effectively zero (< 1e-12).
- `margin_mw` can be negative (base point already overloaded).
- NaN and inf are serialised as JSON numbers (not strings).
- Schema version is currently `3`.

---

### 2.2 Monte Carlo Verification JSON

#### Purpose

Output of the `monte-carlo` command.  Verifies the computed stability
radius certificate by Monte Carlo sampling.

#### Producer

**Module:** `src/stability_radius/verification/monte_carlo.py`

**Function:** `run_monte_carlo_verification()`

**CLI command:** `power_stability_radius monte-carlo`

Output location: `<runs_dir>/<run_id>/monte_carlo_stats.json`

#### Schema

The output is the JSON serialisation of `VerificationResult.to_dict()`:

```json
{
  "schema_version": 1,
  "inputs": {
    "case_id": "case30",
    "results_path": "/path/to/results.json",
    "input_case_path": "/path/to/pglib_opf_case30_ieee.m",
    "slack_bus": 0,
    "n_bus": 30,
    "n_line": 41,
    "dim_balance": 29,
    "n_samples": 50000,
    "seed": 42,
    "chunk_size": 256,
    "sigma_mw": 1.0
  },
  "base_point": {
    "status": "BASE_OK",
    "violated_lines": 0,
    "max_violation_mw": 0.0
  },
  "radius": {
    "status": "RADIUS_OK",
    "r_star": 12.345,
    "argmin_line_pos": 5,
    "argmin_line_idx": 7,
    "min_margin_mw": 3.456,
    "argmin_margin_mw": 3.456,
    "argmin_norm_g": 0.280
  },
  "soundness": {
    "status": "SOUND_PASS",
    "n_ball_samples": 5000,
    "violation_samples": 0,
    "max_violation_mw": 0.0,
    "max_violation_line_idx": -1,
    "tol_mw": 1e-6
  },
  "probabilistic": {
    "status": "PROB_OK",
    "p_safe_gaussian_percent": 99.82,
    "p_safe_gaussian_ci95_low_percent": 99.78,
    "p_safe_gaussian_ci95_high_percent": 99.86,
    "p_ball_analytic_percent": 0.15,
    "p_ball_mc_percent": 0.14,
    "p_ball_mc_ci95_low_percent": 0.11,
    "p_ball_mc_ci95_high_percent": 0.17,
    "eta_safe_given_in_ball_percent": 100.0,
    "eta_ci95_low_percent": 100.0,
    "eta_ci95_high_percent": 100.0,
    "rho": 3.21
  },
  "comparisons": {
    "dc_violation_count": 0,
    "dc_n_samples": 50000,
    "per_line_overload_fractions_conditional_on_pf_converged": { "line_0": 0.0, "line_1": 0.001 },
    "per_line_overload_fraction_denominator": 50000,
    "pf_failure_probability": 0.0,
    "bad_sample_probability": 0.001,
    "pf_failures_gaussian": 0,
    "ac_mc_bound_violations": 0
  },
  "overall": {
    "status": "OK",
    "reasons": []
  }
}
```

#### Status Constants

**Base point:** `BASE_OK`, `BASE_INFEASIBLE`, `BASE_UNKNOWN`

**Radius:** `RADIUS_OK`, `RADIUS_ZERO_BINDING`, `RADIUS_ZERO_BAD_LIMITS`, `RADIUS_INVALID`, `RADIUS_UNKNOWN`

**Soundness:** `SOUND_PASS`, `SOUND_FAIL`, `SOUND_SKIPPED_BASE_INFEASIBLE`, `SOUND_SKIPPED_TRIVIAL_RADIUS`, `SOUND_SKIPPED_INVALID_RADIUS`, `SOUND_SKIPPED_NO_SAMPLES`

**Probabilistic:** `PROB_OK`, `PROB_DEGENERATE_DIMENSION`, `PROB_MC_UNSTABLE`, `PROB_UNKNOWN`

**Overall:** `OK`, `WARN`, `FAIL`

#### Overall Status Contract

- `FAIL` only if `soundness == SOUND_FAIL` or `radius == RADIUS_INVALID`.
- `OK` only if all of: `BASE_OK`, `RADIUS_OK`, `SOUND_PASS`, `PROB_OK`.
- Otherwise `WARN`.

#### Types (Dataclasses)

Defined in `src/stability_radius/verification/types.py`:

- `VerificationInputs`
- `BasePointCheck`
- `RadiusCheck`
- `SoundnessCheck`
- `ProbabilisticCheck`
- `OverallCheck`
- `VerificationResult`

---

### 2.3 Verification Report (Markdown)

#### Purpose

Human-readable multi-case verification report combining DC and AC verification
results with Monte Carlo statistics.

#### Producer

**Module:** `src/stability_radius/verification/generate_report.py`

**Function:** `generate_report_text()`

**CLI command:** `power_stability_radius report`

Output location: configured via `report.io.out` (default: `verification/report.md`)

#### Structure

```markdown
# Verification report

## Setup
- strict: **True**
- n_samples: 50000
- seed: 42
- ...

## case30
- results_path: `/path/to/results.json`
- input_case_path: `/path/to/pglib_opf_case30_ieee.m`
- slack_bus (from results meta): 0

### DC verification
## case30 / DC
- results_status: **ok**
- summary: **OK**
- overall: **OK**

### Inputs
- slack_bus: 0
- n_bus: 30
- ...

### Base point
- status: **BASE_OK**
- violated_lines: 0
- ...

### Radius
- status: **RADIUS_OK**
- r*: 12.345
- ...

### Soundness
- status: **SOUND_PASS**
- ...

### Probabilistic
- status: **PROB_OK**
- p_safe (MC): 99.82%
- ...

### AC verification
(same structure as DC)
```

#### Constraints

- NaN/inf values are rendered as `"n/a"` (never as `"nan%"`).
- When `strict=true` and DC or AC fields are missing, an error is raised.
- A copy of the report is saved in the run directory as `verification_report.md`.

---

### 2.4 h-Vectors (NumPy .npz)

#### Purpose

Store the full-dimensional AC adjoint sensitivity vectors (h-vectors) for all
monitored lines.  These are the injection-space gradients `h = J^{-T} (d|S|/dx)`
and are essential for sigma-radius computation and worst-case analysis.

#### Producer

**Module:** `src/stability_radius/application/cli.py` (via `run_compute()`)

**Function:** Saved when `compute.ac.save_h_vectors=true`

Output location: `<runs_dir>/<run_id>/h_vectors.npz`

#### Schema

NumPy compressed archive (`.npz`) with the following arrays:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `h_from` | `(n_lines, 2*n_bus)` | float64 | h-vectors for the "from" end of each line. Layout: `[h_P_full | h_Q_full]`. |
| `h_to` | `(n_lines, 2*n_bus)` | float64 | h-vectors for the "to" end of each line. |
| `bus_ids` | `(n_bus,)` | int64 | Sorted bus indices (pandapower ordering). |
| `line_ids` | `(n_lines,)` | int64 | Sorted line indices. |

#### Notes

- The h-vectors are expanded from the reduced Jacobian dimension to full bus
  dimension (including a zero entry at the slack bus position).
- For networks with PV buses, the Q-block is scattered to PQ bus positions only;
  PV and slack buses have zero entries in the Q-block.
- Saved via `np.savez_compressed()`.

---

### 2.5 Nonlinear Validation Report

#### Purpose

Store the independent nonlinear AC replay diagnostics produced by
`compute.ac.validation.nonlinear.enabled=true`. The linear AC L2 radius remains
the first-order certificate; this report records how far the worst-case
directions survive under nonlinear pandapower replay.

#### Producer

**Module:** `src/stability_radius/application/cli.py` (via `run_compute()`)

Output locations:

- `<runs_dir>/<run_id>/validation_report.json`
- `<runs_dir>/<run_id>/validation_report.md`

#### JSON Schema

| Key | Type | Description |
|-----|------|-------------|
| `schema_version` | int | Validation report schema version. |
| `case` | string | Case tag/input stem. |
| `top_k_requested` | int | Requested number of replayed AC L2 lines. |
| `top_k_replayed` | int | Number of lines actually replayed. |
| `scale_max` | float | Maximum scale searched beyond the linear boundary. |
| `summary` | object | PF-call counts and gamma distribution summary. |
| `lines` | object[] | Per-line replay summaries and full scale trajectories. |

Per-line entries include `line_id`, `binding_end`, `radius_ac_l2_linear`,
`radius_ac_l2_validated`, `validation_scale_safe`,
`validation_scale_violation`, `nonlinear_conservatism_ratio`,
`pf_replay_status`, `linearization_status`, `max_replay_rel_error`, and
`trajectory`.

---

### 2.6 ASCII Results Tables

#### Purpose

Human-readable formatted tables of per-line results for terminal display
and archival.

#### Producer

**Module:** `src/stability_radius/postprocess/table.py`

**Key functions:**

| Function | Description |
|----------|-------------|
| `format_results_table(results, columns, max_rows)` | Single flat ASCII table. |
| `format_results_table_sections(results, dc_columns, ac_columns, max_rows)` | Two-section (DC + AC) table. |
| `format_radius_summary(results, radius_field)` | One-line statistical summary. |

Output location: `<runs_dir>/<run_id>/results_table.txt`

#### Format

Pipe-delimited, right-aligned numeric columns:

```
line    | flow0_mw | p_limit_mw_est | margin_mw |  norm_g | radius_l2
--------+----------+----------------+-----------+---------+----------
line_0  |   12.345 |          400.0 |   387.655 | 0.28012 |  1383.52
line_1  |  -56.789 |          426.0 |   369.211 | 0.31456 |  1173.83
...
```

In `sections` mode, DC and AC tables are printed with `[DC]` and `[AC]` headers.

#### Summary Line Format

```
Summary(radius_l2): lines=41, finite_radii=41, mean=1234.56, min=12.34, max=9876.54
```

---

### 2.7 CSV Outputs

Several CSV formats are produced by different components:

#### 2.7.1 Per-Line Results CSV

**Producer:** `format_results_csv()` and `format_results_csv_sections()` in
`src/stability_radius/postprocess/table.py`

Output locations:
- `<runs_dir>/<run_id>/results_table_dc.csv` -- DC columns
- `<runs_dir>/<run_id>/results_table_ac.csv` -- AC columns
- `<runs_dir>/<run_id>/results_table.csv` -- flat mode (single file)

**Format:** Standard CSV with header row. First column is `line` (e.g., `line_0`).
Numeric values formatted with `%.6g`. Lines sorted numerically.

```csv
line,flow0_mw,p0_mw,p_limit_mw_est,margin_mw,norm_g,radius_l2
line_0,12.345,12.345,400.0,387.655,0.28012,1383.52
line_1,-56.789,56.789,426.0,369.211,0.31456,1173.83
```

#### 2.7.2 Unified Per-Line Metrics CSV

**Producer:** `entry_points/metrics_analysis.py` (`main()`)

Output location: `<output_dir>/unified_per_line_metrics.csv`

**Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `line_key` | string | `"line_<N>"` |
| `ac_s_limit_mva` | float | Apparent power limit. |
| `s0_binding_mva` | float | Base flow at binding end. |
| `margin_ac_mva` | float | AC margin. |
| `radius_ac_l2` | float | AC L2 radius. |
| `radius_ac_sigma` | float | AC sigma-radius. |
| `radius_ac_metric` | float | AC metric-radius. |
| `sigma_flow_mva` | float | Linearised flow std. |
| `overload_probability_ac` | float | Gaussian overload probability. |
| `loading_ratio` | float | `|S0| / c`. |
| `headroom_mva` | float | `c - |S0|`. |
| `cheb_prob_upper` | float | Cantelli upper bound on overload. |
| `empirical_overload_prob` | float | MC-derived empirical overload fraction. |

#### 2.7.3 Spearman Correlations CSV

**Producer:** `entry_points/metrics_analysis.py`

Output location: `<output_dir>/spearman_correlations.csv`

| Column | Type | Description |
|--------|------|-------------|
| `metric` | string | Metric name (e.g., `"radius_ac_l2"`). |
| `spearman_rho` | float | Spearman rank correlation coefficient. |
| `p_value` | float | Two-sided p-value. |

#### 2.7.4 Precision-at-k CSV

**Producer:** `entry_points/metrics_analysis.py`

Output location: `<output_dir>/precision_at_k.csv`

| Column | Type | Description |
|--------|------|-------------|
| `metric` | string | Metric name. |
| `k` | int | Top-k value (e.g., 3, 5, 10). |
| `mean_empirical_prob` | float | Mean empirical overload probability for top-k lines. |
| `max_empirical_prob` | float | Maximum empirical overload probability for top-k lines. |

#### 2.7.5 Experiment Summary CSV

**Producer:** `src/stability_radius/postprocess/collect_results.py` (`collect()`)

Output location: `run_artifacts/collect_results/all_results.csv`

| Column | Type | Description |
|--------|------|-------------|
| `experiment` | string | Experiment subdirectory name. |
| `case` | string | Case file stem. |
| `input_path` | string | Original input path from `__meta__`. |
| `n_lines` | int | Number of monitored lines. |
| `compute_time_sec` | float | Total computation time. |
| `dc_r_min` | string | Minimum DC L2 radius. |
| `dc_r_median` | string | Median DC L2 radius. |
| `dc_r_mean` | string | Mean DC L2 radius. |
| `dc_r_max` | string | Maximum DC L2 radius. |
| `dc_r_count` | int | Number of finite DC radii. |
| `ac_r_min` | string | Minimum AC L2 radius. |
| `ac_r_median` | string | Median AC L2 radius. |
| `ac_r_mean` | string | Mean AC L2 radius. |
| `ac_r_max` | string | Maximum AC L2 radius. |
| `ac_r_count` | int | Number of finite AC radii. |
| `sigma_r_min` | string | Minimum AC sigma-radius. |
| `sigma_r_median` | string | Median AC sigma-radius. |
| `sigma_r_mean` | string | Mean AC sigma-radius. |
| `sigma_r_max` | string | Maximum AC sigma-radius. |
| `sigma_r_count` | int | Number of finite sigma radii. |

#### 2.7.6 Sigma-Radius Table CSV

**Producer:** `entry_points/run_sigma_radius.py`

Output location: `<output_dir>/table2_sigma_radius.csv`

Contains per-line sigma-radius results for the top-k tightest lines,
formatted for paper inclusion.

---

### 2.8 Plot Outputs

All plots are generated with matplotlib using the `Agg` backend (no display).

#### 2.8.1 Metrics Analysis Plots

**Producer:** `entry_points/metrics_analysis.py`

| File | Description |
|------|-------------|
| `scatter_<metric>.png` | Scatter plot of each metric (x-axis) vs empirical overload probability (y-axis). One file per metric (e.g., `scatter_radius_ac_l2.png`). |
| `spearman_bar.png` | Horizontal bar chart comparing Spearman correlation coefficients across all metrics. Blue bars for lower-is-more-dangerous metrics, orange for higher-is-more-dangerous. |
| `radius_histograms.png` | Overlaid histograms showing the distribution of `radius_ac_l2`, `radius_ac_sigma`, and `radius_ac_metric`. 30 bins, alpha=0.5. |

DPI: 150 for scatter plots, 150 for bar and histogram charts.

#### 2.8.2 Experiment Plots

**Producer:** `entry_points/run_sigma_radius.py`

| File | Description |
|------|-------------|
| `fig2_l2_vs_sigma.png` / `.pdf` | Scatter plot comparing AC L2 radius vs sigma-radius per line. |
| `fig2b_sigma_heatmap.png` / `.pdf` | Per-bus sigma value heatmap. |
| `topology_sigma_radius.png` / `.pdf` | Network topology graph colored by sigma-radius. Node size proportional to bus sigma, edge color mapped from sigma-radius (tighter = red, looser = blue). |

**Producer:** `entry_points/run_pglib_sweep.py`

| File | Description |
|------|-------------|
| `fig1_dc_vs_ac_radius.png` / `.pdf` | DC vs AC L2 radius comparison scatter across all PGLib cases. |
| `fig_critical_lines.png` / `.pdf` | Top critical lines visualization. |
| `fig_flow_vs_limit.png` / `.pdf` | Base flow vs thermal limit scatter. |
| `fig_violation_scale.png` / `.pdf` | Violation scale analysis. |

**Producer:** `src/stability_radius/postprocess/plot_radius_distribution.py`

| File | Description |
|------|-------------|
| Output varies | Distribution plots for radius values. |

**Producer:** `src/stability_radius/postprocess/plot_sigma_vs_time.py`

| File | Description |
|------|-------------|
| Output varies | Sigma-radius evolution over hourly timesteps. |

**Producer:** `src/stability_radius/postprocess/plot_worst_case_heatmap.py`

| File | Description |
|------|-------------|
| Output varies | Heatmap of worst-case perturbation patterns. |

---

### 2.9 Run Artifacts

Every CLI invocation creates a timestamped run directory (or named directory
when `run_dir_mode=overwrite`).

#### Location

```
run_artifacts/<module>/<timestamp>/    # when run_dir_mode=timestamp
run_artifacts/<module>/<run_name>/     # when run_dir_mode=overwrite
```

#### Contents

| File | Format | Description |
|------|--------|-------------|
| `argv.txt` | Text | The exact CLI arguments used. |
| `config_source.yaml` | YAML | Copy of the YAML config file that was loaded. |
| `config.json` | JSON | Fully-resolved effective configuration (all settings as used). |
| `config.yaml` | YAML | OmegaConf-rendered version of effective config. |
| `results.json` | JSON | Computation results (see Section 2.1). |
| `results_table.txt` | Text | ASCII formatted results table. |
| `results_table_dc.csv` | CSV | DC results table (sections mode). |
| `results_table_ac.csv` | CSV | AC results table (sections mode). |
| `results_table.csv` | CSV | Flat-mode results table. |
| `h_vectors.npz` | NumPy | h-vectors (when `save_h_vectors=true`). |
| `validation_report.json` | JSON | Nonlinear AC replay report (when enabled). |
| `validation_report.md` | Markdown | Compact nonlinear AC replay summary (when enabled). |
| `monte_carlo_stats.json` | JSON | Monte Carlo verification output. |
| `verification_report.md` | Markdown | Verification report (report command). |
| `debug.log` | Text | Detailed timestamped log (level=DEBUG). |

**Producer:** `src/stability_radius/application/cli.py` (`_write_run_artifacts()`)

---

### 2.10 Experiment Outputs

#### 2.10.1 PGLib Sweep (`run_pglib_sweep.py`)

Output directory: configured via `output_dir` in YAML (e.g., `run_artifacts/run_pglib_sweep/test_run_6/`)

| File | Format | Description |
|------|--------|-------------|
| `<case_name>/results.json` | JSON | Per-case results (same schema as Section 2.1). |
| `summary.json` | JSON | Aggregated results across all cases. Per case: `n_bus`, `n_line`, `dc_radius_l2_min`, `ac_radius_l2_min`, `compute_time_sec`, etc. |
| `fig1_dc_vs_ac_radius.png` | PNG | DC vs AC radius comparison plot. |

#### 2.10.2 Sigma-Radius Experiment (`run_sigma_radius.py`)

Output directory: configured via `output_dir` in YAML (e.g., `run_artifacts/run_sigma_radius/sigma_radius_hourly/`)

| File | Format | Description |
|------|--------|-------------|
| `results.json` | JSON | Full AC results at the average operating point. |
| `sigma_arrays.json` | JSON | Per-bus sigma values (serialised from UC.jl or synthetic). |
| `h_vectors.npz` | NumPy | Saved h-vectors. |
| `table2_sigma_radius.csv` | CSV | Formatted sigma-radius table for paper inclusion. |
| `fig2_l2_vs_sigma.png` | PNG | L2 vs sigma-radius scatter. |
| `fig2b_sigma_heatmap.png` | PNG | Per-bus sigma heatmap. |
| `topology_sigma_radius.png` | PNG | Network graph colored by sigma-radius. |
| `validation.json` | JSON | MC validation results. |
| `verification_results.json` | JSON | Per-line nonlinear worst-case verification. |

#### 2.10.3 Worst-Case Verification (`run_worst_case_verify.py`)

| File | Format | Description |
|------|--------|-------------|
| `*_worst_case.json` | JSON | Worst-case verification results per line. |

#### 2.10.4 Scalability (`run_scalability.py`)

| File | Format | Description |
|------|--------|-------------|
| `scalability.json` | JSON | Timing results for different case sizes. |

---

### 2.11 Debug Logs

#### Purpose

Detailed computation logs for debugging and reproducibility auditing.

#### Location

`<runs_dir>/<run_id>/debug.log` (or wherever the logging framework writes)

#### Format

Standard Python logging format:

```
2026-03-10 14:23:45,123 stability_radius.workflows INFO  pglib_opf_case30_ieee: Read Data
2026-03-10 14:23:45,456 stability_radius.workflows DEBUG Resolved path: data/input/... -> /absolute/...
```

Log levels:
- Console: controlled by `logging.level_console` (default: `INFO`)
- File: controlled by `logging.level_file` (default: `DEBUG`)

---

## Units Contract

The project enforces a consistent units convention across all data formats:

| Quantity | Unit | Notes |
|----------|------|-------|
| Active power (P, f0, margin, radius) | MW | Megawatts |
| Reactive power (Q) | MVAr | Megavolt-amperes reactive |
| Apparent power (S, limit) | MVA | Megavolt-amperes |
| Voltage magnitude | p.u. | Per-unit |
| Voltage angle | radians (internal) / degrees (MATPOWER) | Conversion at parse time |
| Resistance, reactance, susceptance | p.u. | Per-unit on `baseMVA` |
| Thermal ratings (rateA, sn_mva) | MVA | Used as MW under DC lossless (PF=1) assumption |
| Sigma (std deviation) | MW / MVAr | Matches the power type being perturbed |
| Probability | [0, 1] | Dimensionless |
| Radius (L2) | MW | Euclidean distance in injection space |
| Radius (sigma) | dimensionless | Normalised by sigma |

---

## Internal Data Structures

These are not file formats but are important in-memory structures that bridge
parsing and output.

### `LineBaseQuantities` (dataclass)

**Module:** `src/stability_radius/radii/common.py`

Container for per-line base quantities used throughout radius calculations.

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `line_indices` | `list[int]` | `(m,)` | Sorted pandapower line indices. |
| `flow0_mw` | `np.ndarray` | `(m,)` | Signed base flows (MW). |
| `p0_abs_mw` | `np.ndarray` | `(m,)` | Absolute base flows (MW). |
| `limit_mva_assumed_mw` | `np.ndarray` | `(m,)` | Thermal limits (MVA as MW). |
| `margin_mw` | `np.ndarray` | `(m,)` | Margins: `limit - |flow0|`. |
| `is_unconstrained` | `np.ndarray\|None` | `(m,)` bool | True for unconstrained lines. |
| `opf_status` | `str\|None` | -- | OPF solver status. |
| `opf_objective` | `float\|None` | -- | OPF objective value. |
| `bus_ids` | `list[int]\|None` | `(n,)` | Sorted bus indices. |
| `bus_injections_mw` | `np.ndarray\|None` | `(n,)` | Per-bus net injection (MW). |
| `opf_gen_dispatch_mw_by_name` | `tuple[tuple[str,float],...]\|None` | -- | Generator dispatch keyed by PyPSA name. |
| `opf_ext_grid_absorption_mw` | `float` | -- | Slack bus absorption (MW). |

### `DCExtensionsConfig` / `ACExtensionsConfig` (dataclasses)

**Module:** `src/stability_radius/workflows.py`

Configuration holders for optional post-processing extensions (DC probabilistic,
N-1, AC sigma/metric radius, h-vector saving).

### `VerificationResult` (dataclass)

**Module:** `src/stability_radius/verification/types.py`

Structured verification result composed of `VerificationInputs`,
`BasePointCheck`, `RadiusCheck`, `SoundnessCheck`, `ProbabilisticCheck`,
`OverallCheck`, and a free-form `comparisons` dict.  Serialised to JSON
via `.to_dict()` (which calls `dataclasses.asdict()`).
