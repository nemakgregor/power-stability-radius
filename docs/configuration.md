# Configuration Reference

This document describes the full configuration system for the
**power-stability-radius** project: YAML files, dataclass defaults,
the `extends` inheritance mechanism, CLI flag interaction, and
performance-critical parameters.

---

## Table of Contents

1. [Architecture overview](#architecture-overview)
2. [The `extends` inheritance mechanism](#the-extends-inheritance-mechanism)
3. [Configuration file inventory](#configuration-file-inventory)
4. [CLI-to-YAML interaction](#cli-to-yaml-interaction)
5. [Dataclass reference (Python defaults)](#dataclass-reference-python-defaults)
6. [Parameter reference by section](#parameter-reference-by-section)
   - [Logging](#logging)
   - [IO / Reproducibility](#io--reproducibility)
   - [OPF (DC Optimal Power Flow)](#opf-dc-optimal-power-flow)
   - [HiGHS solver](#highs-solver)
   - [DC model](#dc-model)
   - [DC extensions (probabilistic, N-1)](#dc-extensions-probabilistic-n-1)
   - [AC model](#ac-model)
   - [AC Feasibility Power Flow (AC FPF)](#ac-feasibility-power-flow-ac-fpf)
   - [AC extensions (sigma-radius, metric-radius)](#ac-extensions-sigma-radius-metric-radius)
   - [Monte Carlo verification](#monte-carlo-verification)
   - [Report generation](#report-generation)
   - [Table formatting](#table-formatting)
   - [Tolerances](#tolerances)
   - [Compute outputs](#compute-outputs)
7. [Experiment configurations](#experiment-configurations)
8. [Full example configs](#full-example-configs)
9. [Performance-critical parameters](#performance-critical-parameters)
10. [Reproducibility contract](#reproducibility-contract)

---

## Architecture overview

Configuration is resolved from three layers, from lowest to highest
priority:

```
Python dataclass defaults  (src/stability_radius/config.py)
        |
        v
  YAML config files        (conf/*.yaml, conf/experiments/*.yaml)
        |
        v
  CLI flags                (--dc-mode, --ac-chunk-size, ...)
```

The YAML layer uses OmegaConf (from `hydra-core`) for loading and
merging.  The Python layer defines frozen `@dataclass` objects that
serve as programmatic defaults.  CLI flags are parsed with `argparse`
and always take the highest priority, overriding both YAML and
dataclass values.

### Determinism contract

Some parameters must be identical between the Python defaults and the
YAML shared defaults.  If they diverge, CI tests (which use Python
defaults) and CLI experiments (which use YAML) can silently compute
different results.  These parameters are marked with
**[determinism-critical]** throughout this document.

---

## The `extends` inheritance mechanism

YAML config files may declare an `extends` key that references one or
more parent config files.  The loader in `config.py`
(`load_project_config` / `_load_with_extends`) resolves these
recursively and merges them using `OmegaConf.merge`.

### Single parent

```yaml
# conf/experiments/case30.yaml
extends: ../config.yaml

compute:
  input: data/input/pglib_opf_case30_ieee.m
  slack_bus: 0
```

The child file inherits all keys from `config.yaml` and overrides only
the keys it declares.

### Multiple parents

```yaml
# conf/config.yaml
extends:
  - ./config_shared.yaml
  - ./config_compute.yaml
  - ./config_monte_carlo.yaml
  - ./config_report.yaml
```

Parents listed later take precedence over earlier ones.  The local
file (the one containing the `extends` key) takes the highest
precedence of all.

### Resolution rules

| Rule | Behavior |
|------|----------|
| Paths | Resolved relative to the **parent file's directory**, not CWD |
| Cycles | Detected and raise `ValueError` |
| Missing parents | Raise `FileNotFoundError` |
| Depth | Unlimited nesting (parent can itself extend another parent) |
| Merge semantics | OmegaConf deep merge: mappings are merged recursively, scalars and lists are replaced |

### Internal implementation

```python
def _load_with_extends(path: Path, *, stack: tuple[Path, ...]) -> Any:
    # 1. Load the file with OmegaConf.load()
    # 2. Extract and normalize the "extends" value to a list
    # 3. Recursively load each parent (cycle detection via `stack`)
    # 4. Strip the "extends" key from the local config
    # 5. Return OmegaConf.merge(*parents, local)
```

---

## Configuration file inventory

### Core configs (`conf/`)

| File | Purpose |
|------|---------|
| `config.yaml` | Main entry point. Extends `config_shared`, `config_compute`, `config_monte_carlo`, and `config_report`. Contains no parameters of its own -- it is purely a composition hub. |
| `config_shared.yaml` | Shared defaults used by every command: IO policy, logging, AC PF backend, OPF solver settings, DC model defaults, tolerances, and table formatting. |
| `config_compute.yaml` | Parameters for the `compute` command: input file, slack bus, base dispatch mode, DC/AC compute flags, AC FPF settings, sigma-radius settings, and output options. |
| `config_dc_extensions.yaml` | Optional DC post-processing: probabilistic sigma-radius and N-1 contingency analysis. Intentionally separate from the main compute config to keep defaults compact. |
| `config_monte_carlo.yaml` | Parameters for the `monte-carlo` verification command: sampling, tolerances, DC/AC sigma overrides. |
| `config_report.yaml` | Parameters for the `report` command: IO paths, sampling, tolerances, case list, and plot generation. |

### Experiment configs (`conf/experiments/`)

| File | Purpose |
|------|---------|
| `case30.yaml` | Single-case experiment for IEEE 30-bus. Extends `config.yaml`. |
| `case118.yaml` | Single-case experiment for IEEE 118-bus. Extends `config.yaml`. |

### Research experiment configs (`experiments/configs/`)

| File | Purpose |
|------|---------|
| `pglib_sweep.yaml` | Multi-case sweep across PGLib networks (IEEE, ACTIVSg, RTE, Polish, PEGASE). Each entry can override `base_dispatch`, `headroom_factor`, `timeout`, and AC settings. |
| `sigma_case2000_goc.yaml` | Sigma-radius experiment for 2000-bus GOC case with synthetic sigma (10% of load). |
| `sigma_case2736sp_k.yaml` | Sigma-radius experiment for 2736-bus Polish case with synthetic sigma. |
| `sigma_case2869_pegase.yaml` | Sigma-radius experiment for 2869-bus PEGASE case with synthetic sigma. |
| `uc_jl_case118.yaml` | Multi-hour AC sigma-radius experiment using UnitCommitment.jl time-series data for case118. |

---

## CLI-to-YAML interaction

The CLI (`power_stability_radius`) uses a two-pass argument parsing
strategy:

1. **Pre-parse**: Extract `--config` path (default: `conf/config.yaml`)
   before any subcommand parsing.
2. **Load YAML**: Call `load_project_config(path)` to build the
   composed config tree via `extends`.
3. **Build parser**: Construct `argparse` with YAML values as defaults
   using `_cfg_get(cfg, "dotted.key", python_default)`.
4. **Parse CLI**: CLI flags override YAML defaults.

### Lookup order for each parameter

```
CLI flag  >  YAML value (via extends chain)  >  Python dataclass default
```

### Example: `--dc-chunk-size`

```python
p_compute.add_argument(
    "--dc-chunk-size",
    type=int,
    default=int(_cfg_get(cfg, "dc.chunk_size", DEFAULT_DC.chunk_size)),
    #                        ^^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^^^
    #                        YAML key path     Python default (256)
)
```

If the user passes `--dc-chunk-size 64`, that value wins.  Otherwise
the YAML `dc.chunk_size` is used.  If the YAML key is missing, the
Python `DCConfig.chunk_size` default (256) is used.

### Subcommands

| Subcommand | Description |
|------------|-------------|
| `compute` (alias: `demo`) | Compute DC and/or AC stability radii for a MATPOWER case |
| `monte-carlo` | Run Monte Carlo verification against computed results |
| `report` | Generate a verification report across multiple cases |
| `table` | Print or export a results table from a JSON file |

---

## Dataclass reference (Python defaults)

These frozen dataclasses in `src/stability_radius/config.py` provide the
programmatic defaults when YAML values are missing.

### `LoggingConfig`

```python
@dataclass(frozen=True)
class LoggingConfig:
    runs_dir: str = "runs"
    module_name: str = "general"
    level_console: str = "INFO"
    level_file: str = "DEBUG"
    run_dir_mode: str = "timestamp"  # "timestamp" | "overwrite"
    run_name: str = "latest"  # used only when run_dir_mode="overwrite"
```

### `HiGHSConfig`

```python
@dataclass(frozen=True)
class HiGHSConfig:
    solver_name: str = "highs"
    threads: int = 4
    random_seed: int = 42
    user_objective_scale: int = -1
    user_bound_scale: int = -10
    primal_feasibility_tolerance: float = 1e-9
    dual_feasibility_tolerance: float = 1e-9
```

### `OPFConfig`

```python
@dataclass(frozen=True)
class OPFConfig:
    highs: HiGHSConfig = field(default_factory=HiGHSConfig)
    unconstrained_line_nom_mw: float = 1e5  # [determinism-critical]
    headroom_factor: float = 0.98  # [determinism-critical]
    ext_grid_marginal_cost_base: float = 1000.0
```

### `DCConfig`

```python
@dataclass(frozen=True)
class DCConfig:
    mode: str = "operator"  # "operator" | "materialize"
    chunk_size: int = 256
    dtype: str = "float64"  # "float64" | "float32"
```

### `MonteCarloConfig`

```python
@dataclass(frozen=True)
class MonteCarloConfig:
    n_samples: int = 50_000
    seed: int = 42
    chunk_size: int = 256
    feas_tol_mw: float = 0.0
    cert_tol_mw: float = 1e-6
    cert_max_samples: int = 5_000
```

### `DCExtensionsConfig` (in `workflows.py`)

```python
@dataclass(frozen=True)
class DCExtensionsConfig:
    probabilistic_enabled: bool = False
    nminus1_enabled: bool = False
    nminus1_update_sensitivities: bool = True
    nminus1_islanding: str = "skip"  # "skip" | "raise"
```

### `ACExtensionsConfig` (in `workflows.py`)

```python
@dataclass(frozen=True)
class ACExtensionsConfig:
    sigma_p_mw_source: str = ""  # "uniform" | "uc_jl" | ""
    sigma_q_mvar_source: str = ""  # "uniform" | "uc_jl" | ""
    sigma_p_mw_uniform: float = 1.0
    sigma_q_mvar_uniform: float = 1.0
    sigma_p_mw_array: np.ndarray | None = None
    sigma_q_mvar_array: np.ndarray | None = None
    sigma_n_timesteps: int | None = None
    metric_enabled: bool = False
    save_h_vectors: bool = False
    nonlinear_validation_enabled: bool = False
    nonlinear_validation_top_k: int = 20
    nonlinear_validation_scale_max: float = 5.0
    nonlinear_validation_tol: float = 0.01
    nonlinear_validation_max_iter: int = 20
```

### `ACFPFConfig` (in `base_point/pandapower_opp.py`)

```python
@dataclass(frozen=True)
class ACFPFConfig:
    pg0_source: str = "case"  # "case" | "midpoint"
    vm_min_pu: float = 0.9
    vm_max_pu: float = 1.1
    max_iteration: int = 300
    max_loading_percent: float = 99.0
    pdipm_feastol: float = 1e-4
    pdipm_gradtol: float = 1e-4
    pdipm_comptol: float = 1e-4
    pdipm_costtol: float = 1e-4
    opf_violation: float = 1e-4
    init: str = "dc"  # "dc" | "flat"
    max_attempts: int = 1
    per_attempt_timeout: float = 0  # seconds; 0 = unlimited
```

---

## Parameter reference by section

### Logging

| Parameter | YAML key | CLI flag | Type | Default | Description |
|-----------|----------|----------|------|---------|-------------|
| Runs directory | `logging.runs_dir` | `--runs-dir` | str | `"runs"` | Base directory for run output folders |
| Console log level | `logging.level_console` | `--log-level` | str | `"INFO"` | Python logging level for console output |
| File log level | `logging.level_file` | `--log-file-level` | str | `"DEBUG"` | Python logging level for the run log file |
| Run directory mode | `logging.run_dir_mode` | `--run-dir-mode` | str | `"timestamp"` | `"timestamp"`: create `run_artifacts/<module>/<timestamp>/`. `"overwrite"`: create/recreate `run_artifacts/<module>/<run_name>/` |
| Run name | `logging.run_name` | `--run-name` | str | `"latest"` | Folder name when `run_dir_mode="overwrite"` |
| Run tests | `run_tests` | `--run-tests` | int | `1` | If 1, run the test suite before the main command |

### IO / Reproducibility

| Parameter | YAML key | CLI flag | Type | Default | Description |
|-----------|----------|----------|------|---------|-------------|
| Allow download | `io.allow_download` | `--allow-download` | int | `0` | If 1, automatically download supported PGLib `.m` files from GitHub when the local file is missing. If 0, fail fast on missing input. |

### OPF (DC Optimal Power Flow)

These settings control the DC OPF dispatch solver (PyPSA + HiGHS).
The DC OPF is used only when `base_dispatch=dc_opf`.

| Parameter | YAML key | CLI flag | Type | Default | Effect | Notes |
|-----------|----------|----------|------|---------|--------|-------|
| Solver name | `opf.solver_name` | `--opf-solver-name` | str | `"highs"` | Must be `"highs"`. Project policy. | Do not change. |
| Threads | `opf.threads` | `--opf-threads` | int | `4` | Number of HiGHS solver threads. | Performance-oriented default. Use `1` for the strictest reproducibility. |
| Random seed | `opf.random_seed` | `--opf-random-seed` | int | `42` | HiGHS random seed for tie-breaking. | **[determinism-critical]** |
| Headroom factor | `opf.headroom_factor` | `--opf-headroom-factor` | float | `0.98` | Fraction of thermal capacity used as OPF line constraint. `0.98` means 2% security margin. | **[determinism-critical]** Valid range: (0, 1]. Values below ~0.90 can cause OPF infeasibility on tight networks. |
| Unconstrained line limit | `opf.unconstrained_line_nom_mw` | `--opf-unconstrained-line-nom-mw` | float | `1e5` | Surrogate finite thermal limit (MW) for lines with `rateA=0/inf/NaN`. PyPSA requires finite limits. | **[determinism-critical]** Must match `DEFAULT_UNCONSTRAINED_LINE_NOM_MW` in Python. |
| Ext grid marginal cost | `opf.ext_grid_marginal_cost_base` | `--opf-ext-grid-marginal-cost-base` | float | `1000.0` | Cost assigned to the external grid "generator" in PyPSA DC OPF. Must be large enough to represent slack feasibility, but not so large as to cause LP scaling issues. | Typical range: 100--10000. |

### HiGHS solver

Advanced HiGHS parameters are available only via the `HiGHSConfig`
dataclass (not exposed as YAML keys or CLI flags):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_objective_scale` | int | `-1` | Objective scaling exponent |
| `user_bound_scale` | int | `-10` | Bound scaling exponent |
| `primal_feasibility_tolerance` | float | `1e-9` | Primal feasibility tolerance |
| `dual_feasibility_tolerance` | float | `1e-9` | Dual feasibility tolerance |

### DC model

| Parameter | YAML key | CLI flag | Type | Default | Effect |
|-----------|----------|----------|------|---------|--------|
| Mode | `dc.mode` | `--dc-mode` | str | `"operator"` | `"operator"`: compute H*x products on-the-fly (fast, low memory, but no N-1). `"materialize"`: build the full dense H matrix (high memory, required for N-1 contingency). |
| Chunk size | `dc.chunk_size` | `--dc-chunk-size` | int | `256` | Number of lines per LU-solve batch. Larger values use more memory but reduce Python loop overhead. | **[performance-critical]**
| Dtype | `dc.dtype` | `--dc-dtype` | str | `"float64"` | Floating-point precision. `"float32"` halves memory at the cost of ~7 significant digits. Use `"float64"` for research-grade results. |
| Injection std | `compute.dc.inj_std_mw` | `--inj-std-mw` | float | `1.0` | Standard deviation (MW) for the Gaussian injection distribution used in DC Monte Carlo and DC sigma-radius. Stored in results metadata. |

### DC extensions (probabilistic, N-1)

These are advanced options, disabled by default. Enable via CLI flags
or by adding `config_dc_extensions.yaml` to your experiment's
`extends` list.

| Parameter | YAML key | CLI flag | Type | Default | Description |
|-----------|----------|----------|------|---------|-------------|
| Probabilistic enabled | `compute.dc.probabilistic.enabled` | `--compute-dc-probabilistic` | int | `0` | If 1, compute DC sigma-radius (`radius_sigma`) and overload probability as post-processing. |
| N-1 enabled | `compute.dc.nminus1.enabled` | `--compute-nminus1` | int | `0` | If 1, compute effective N-1 DC radii for single-line outage contingencies. Requires `--dc-mode materialize`. |
| N-1 update sensitivities | `compute.dc.nminus1.update_sensitivities` | `--nminus1-update-sensitivities` | int | `1` | If 1, use Woodbury/LODF-based sensitivity update for each contingency (more accurate, slower). If 0, use the base-case sensitivities. |
| N-1 islanding | `compute.dc.nminus1.islanding` | `--nminus1-islanding` | str | `"skip"` | `"skip"`: silently skip contingencies that cause network islanding. `"raise"`: raise an error on islanding. |

### AC model

| Parameter | YAML key | CLI flag | Type | Default | Description |
|-----------|----------|----------|------|---------|-------------|
| Compute AC | `compute.ac.compute` | `--compute-ac` | int | `1` | If 1, compute AC stability radii. |
| Chunk size | `compute.ac.chunk_size` | `--ac-chunk-size` | int | `256` | Lines per LU-solve batch for AC sensitivity computation. | **[performance-critical]**
| Balance | `compute.ac.balance` | `--ac-balance` | int | `1` | If 1, enforce balanced (zero-sum) disturbance directions in the AC certificate. |
| PF solver | `ac.pf_solver` | `--ac-pf-solver` | str | `"pandapower"` | AC power flow backend. `"pandapower"` or `"pypsa"`. AC Monte Carlo currently only supports pandapower. |
| PF init | `compute.ac.pf_init` | `--ac-pf-init` | str | `"flat"` | AC PF initialization strategy. `"flat"`: flat start. `"dc"`: DC theta angles as initial guess. `"pp"`: run pandapower PF first for an explicit initial guess. |
| Lossless | `ac.lossless` | `--ac-lossless` | int | `1` | If 1, enforce the supported lossless series-only model for both PF and Jacobian. `lossless=false` is fail-fast in certificate mode until the full pi/shunt Jacobian is implemented. |
| Distributed slack | `compute.ac.distributed_slack` | N/A | bool | `true` (sweep) | Distribute active power losses proportionally to headroom (Pmax - Pset). Disable on networks where pandapower crashes (e.g., large GOC/PEGASE cases). |
| Transformer model | `compute.ac.trafo_model` | N/A | str | `"pi"` | Transformer equivalent circuit model type. |
| Save h-vectors | `compute.ac.save_h_vectors` | `--ac-save-h-vectors` | int | `0` | If 1, save full adjoint sensitivity h-vectors to a `.npz` file alongside results. |
| Nonlinear validation | `compute.ac.validation.nonlinear.enabled` | `--ac-validate-nonlinear` | int | `0` | If 1, replay nonlinear pandapower PF for the top-k smallest finite AC L2 radii. |
| Validation top-k | `compute.ac.validation.nonlinear.top_k` | `--ac-validation-top-k` | int | `20` | Number of AC L2 lines to replay when nonlinear validation is enabled. |
| Validation scale max | `compute.ac.validation.nonlinear.scale_max` | `--ac-validation-scale-max` | float | `5.0` | Maximum scale searched relative to the linear boundary. |
| Validation tolerance | `compute.ac.validation.nonlinear.tol` | `--ac-validation-tol` | float | `0.01` | Binary-search tolerance on the replay scale. |
| Validation max iterations | `compute.ac.validation.nonlinear.max_iter` | `--ac-validation-max-iter` | int | `20` | Maximum violation-scale binary-search iterations. |

### AC Feasibility Power Flow (AC FPF)

Used when `base_dispatch=ac_fpf`.  The AC FPF solves an AC OPF via
pandapower's `runopp()` (PDIPM interior-point solver) to find a
feasible operating point.

| Parameter | YAML key | CLI flag | Type | Default | Valid range | Description |
|-----------|----------|----------|------|---------|-------------|-------------|
| Pg0 source | `compute.ac_fpf.pg0_source` | `--ac-fpf-pg0-source` | str | `"case"` | `"case"`, `"midpoint"` | Initial dispatch guess. `"case"`: use `net.gen.p_mw` from the input case. `"midpoint"`: use `(min_p_mw + max_p_mw) / 2`. |
| Vm min | `compute.ac_fpf.vm_min_pu` | N/A | float | `0.9` | [0.8, 1.0] | Minimum bus voltage magnitude (p.u.) for OPF constraint. Relaxed to 0.85 then 0.80 on retry attempts. |
| Vm max | `compute.ac_fpf.vm_max_pu` | N/A | float | `1.1` | [1.0, 1.2] | Maximum bus voltage magnitude (p.u.) for OPF constraint. Relaxed to 1.15 then 1.20 on retry attempts. |
| Max iterations | `compute.ac_fpf.max_iteration` | N/A | int | `300` | [50, 1000] | Maximum PDIPM interior-point iterations. Increase for hard-to-converge networks. |
| Max loading % | `compute.ac_fpf.max_loading_percent` | N/A | float | `99.0` | [80, 100] | Line loading limit for OPF (%). Set below 100 to compensate for PDIPM solver tolerance, ensuring the solution satisfies the true 100% limit. | **[performance-critical]**
| Init | `compute.ac_fpf.init` | N/A | str | `"dc"` | `"dc"`, `"flat"` | Power flow initialization for `runopp()`. `"dc"` provides a warm start from DC angles; `"flat"` starts from V=1, theta=0. Use `"flat"` when DC init causes convergence issues (e.g., networks with many transformers). |
| Max attempts | `compute.ac_fpf.max_attempts` | N/A | int | `1` | [1, 3] | Number of bounded `runopp()` attempts before giving up. Attempt 1 uses configured bounds; attempt 2 widens to [0.85, 1.15]; attempt 3 widens to [0.80, 1.20]. If all attempts fail, the AC FPF run fails. |
| Per-attempt timeout | `compute.ac_fpf.per_attempt_timeout` | N/A | float | `0` | >= 0 | Timeout in seconds for each `runopp()` call. `0` means no timeout. A positive value (e.g., 180) prevents a single slow attempt from exhausting the subprocess timeout. | **[performance-critical]**
| PDIPM feastol | `compute.ac_fpf.pdipm_feastol` | N/A | float | `1e-4` | (0, 1) | Feasibility (equality constraint) tolerance. `0` uses the OPF_VIOLATION default. |
| PDIPM gradtol | `compute.ac_fpf.pdipm_gradtol` | N/A | float | `1e-4` | (0, 1) | Gradient (optimality) tolerance. |
| PDIPM comptol | `compute.ac_fpf.pdipm_comptol` | N/A | float | `1e-4` | (0, 1) | Complementary slackness (inequality) tolerance. |
| PDIPM costtol | `compute.ac_fpf.pdipm_costtol` | N/A | float | `1e-4` | (0, 1) | Cost (objective convergence) tolerance. |
| OPF violation | `compute.ac_fpf.opf_violation` | N/A | float | `1e-4` | (0, 1) | General constraint violation tolerance. Also used as the default for PDIPM_FEASTOL when feastol is 0. |

### AC extensions (sigma-radius, metric-radius)

These compute probabilistic and metric stability radii for the AC
model.  Disabled by default.

| Parameter | YAML key | CLI flag | Type | Default | Description |
|-----------|----------|----------|------|---------|-------------|
| Sigma P source | `compute.ac.sigma.sigma_p_mw_source` | `--ac-sigma-p-source` | str | `""` | `""`: disabled. `"uniform"`: use `sigma_p_mw_uniform` for all buses. `"uc_jl"`: use per-bus arrays from UnitCommitment.jl data. |
| Sigma Q source | `compute.ac.sigma.sigma_q_mvar_source` | `--ac-sigma-q-source` | str | `""` | Same options as sigma P source, for reactive power. |
| Sigma P uniform | `compute.ac.sigma.sigma_p_mw_uniform` | `--ac-sigma-p-uniform` | float | `1.0` | Per-bus standard deviation for active power (MW) when source is `"uniform"`. |
| Sigma Q uniform | `compute.ac.sigma.sigma_q_mvar_uniform` | `--ac-sigma-q-uniform` | float | `1.0` | Per-bus standard deviation for reactive power (Mvar) when source is `"uniform"`. |
| Metric enabled | `compute.ac.metric.enabled` | `--ac-metric-enabled` | int | `0` | If 1, also compute metric-radius with M = diag(1/sigma^2). Serves as a cross-check against sigma-radius. |

### Monte Carlo verification

Parameters for the `monte-carlo` subcommand, which verifies computed
stability certificates via random sampling.

| Parameter | YAML key | CLI flag | Type | Default | Description |
|-----------|----------|----------|------|---------|-------------|
| Mode | `monte_carlo.mode` | `--mode` | str | `"dc"` | `"dc"`: DC certificate verification. `"ac"`: AC certificate verification (runs AC PF per sample). |
| Results path | `monte_carlo.results` | `--results` | str | `""` | Path to the `results.json` file to verify. Required. |
| Input path | `monte_carlo.input` | `--input` | str | `""` | Path to the MATPOWER `.m` input case. Required. |
| Slack bus | `monte_carlo.slack_bus` | `--slack-bus` | int | `0` | Slack bus index. Falls back to `compute.slack_bus` if unset. |
| N samples | `monte_carlo.sampling.n_samples` | `--n-samples` | int | `50000` | Number of Monte Carlo random samples. | **[performance-critical]**
| Seed | `monte_carlo.sampling.seed` | `--seed` | int | `42` | Random seed for reproducibility. | **[determinism-critical]**
| Chunk size | `monte_carlo.sampling.chunk_size` | `--chunk-size` | int | `256` | Samples per batch. |
| Feasibility tol | `monte_carlo.tolerances.feas_tol` | `--feas-tol` | float | `0.0` | Tolerance (MW for DC, MVA for AC) for classifying a sample as feasible. `0.0` means exact. |
| Certificate tol | `monte_carlo.tolerances.cert_tol` | `--cert-tol` | float | `1.0` | Tolerance for certificate verification. Samples within this tolerance of the boundary are counted separately. |
| Cert max samples | `monte_carlo.tolerances.cert_max_samples` | `--cert-max-samples` | int | `5000` | Maximum number of samples to use for certificate tightness estimation. |
| DC sigma override | `monte_carlo.dc.sigma_override_mw` | `--sigma-override-mw` | float | `null` | Override the Gaussian sigma (MW) for DC MC sampling. If null, uses `results.__meta__.dc.inj_std_mw`. |
| AC sigma P | `monte_carlo.ac.sigma_p_mw` | `--ac-sigma-p-mw` | float | `1.0` | Gaussian standard deviation for active power perturbations (MW) in AC MC. |
| AC sigma Q | `monte_carlo.ac.sigma_q_mvar` | `--ac-sigma-q-mvar` | float | `1.0` | Gaussian standard deviation for reactive power perturbations (Mvar) in AC MC. |
| AC PF solver | `ac.pf_solver` | `--ac-pf-solver` | str | `"pandapower"` | AC power flow backend for per-sample evaluation. Only `"pandapower"` is supported for AC MC. |
| AC lossless | `ac.lossless` | `--ac-lossless` | int | `1` | Lossless series-only model flag. `0` is fail-fast for certificate mode until the full AC pi/shunt Jacobian is implemented. |
| AC basepoint tol | `ac.basepoint_s_tol_mva` | `--ac-basepoint-s-tol-mva` | float | `1e-3` | Tolerance (MVA) for base-point consistency check. Compares |S| at both ends of every monitored line against results.json base fields. |

### Report generation

Parameters for the `report` subcommand, which generates a verification
report across multiple cases.

| Parameter | YAML key | CLI flag | Type | Default | Description |
|-----------|----------|----------|------|---------|-------------|
| Results dir | `report.io.results_dir` | `--results-dir` | str | `"verification/results"` | Directory containing per-case result JSON files. |
| Output file | `report.io.out` | `--out` | str | `"verification/report.md"` | Output Markdown report path. |
| Strict mode | `report.strict` | `--strict` | int | `0` | If 1, fail if any result file is missing. If 0, skip missing cases with a warning. |
| Generate plots | `report.generate_plots` | N/A | bool | `false` | Whether to generate matplotlib plots in the report. |
| Cases list | `report.cases` | N/A | list | (see YAML) | List of case entries with `id`, `input`, `results`, and `known_critical_pairs`. |

Report also accepts `--n-samples`, `--seed`, `--chunk-size`,
`--feas-tol`, `--cert-tol`, `--cert-max-samples`,
`--sigma-override-mw`, `--ac-sigma-p-mw`, `--ac-sigma-q-mvar`,
`--ac-pf-solver`, `--ac-lossless`, and `--ac-basepoint-s-tol-mva`
with semantics identical to the Monte Carlo subcommand.

### Table formatting

| Parameter | YAML key | CLI flag | Type | Default | Description |
|-----------|----------|----------|------|---------|-------------|
| Format | `table.format` | `--format` | str | `"sections"` | `"sections"`: print separate DC and AC tables. `"flat"`: print a single table with selected columns. |
| Columns | `table.columns` | `--columns` | str/list | `[]` | Base column list (applies to both sections). Empty means use section-specific defaults. |
| DC extra columns | `table.dc_extra_columns` | N/A | list | `[flow0_mw, p0_mw, p_limit_mw_est, margin_mw, norm_g, radius_l2, constraint_status_l2, certificate_radius_l2]` | Additional columns for the DC section. |
| AC extra columns | `table.ac_extra_columns` | N/A | list | `[ac_s_limit_mva, ac_s0_from_mva, ac_s0_to_mva, margin_ac_mva, "||h||2", binding_end, radius_ac_l2, constraint_status_ac_l2, certificate_radius_ac_l2]` | Additional columns for the AC section. |
| Radius field | N/A | `--radius-field` | str | `"radius_l2"` | Field to sort by in flat mode. |
| Max rows | N/A | `--max-rows` | int | `null` | Limit the number of rows displayed. `null` means show all. |

### Tolerances

| Parameter | YAML key | CLI flag | Type | Default | Description |
|-----------|----------|----------|------|---------|-------------|
| OPF-DC flow consistency | `tolerances.opf_dc_flow_consistency_tol_mw` | `--opf-dc-flow-consistency-tol-mw` | float | `1.0` | Maximum allowed deviation (MW) between OPF dispatch and DC flow reconstruction. IEEE cases match to machine precision; PEGASE networks may show ~1-2 MW residuals. |
| OPF bus balance | `tolerances.opf_bus_balance_tol_mw` | `--opf-bus-balance-tol-mw` | float | `1.0` | Maximum allowed bus power balance residual (MW). |

### Compute outputs

| Parameter | YAML key | CLI flag | Type | Default | Description |
|-----------|----------|----------|------|---------|-------------|
| Export results | `compute.output.export_results` | `--export-results` | str | `""` | Path to export results as JSON. Empty means do not export. |
| Save CSV | `compute.output.save_csv` | `--save-csv` | int | `1` | If 1, save a CSV table alongside the run artifacts. |
| Max rows | `compute.output.max_rows` | `--max-rows` | int | `null` | Limit rows in the printed/saved table. |
| Table columns | `compute.output.table_columns` | `--table-columns` | str | `""` | Comma-separated column names for flat table output. Empty uses sectioned defaults. |

---

## Experiment configurations

### PGLib sweep (`experiments/configs/pglib_sweep.yaml`)

This config defines a list of PGLib test cases under the `cases` key.
Each entry can override per-case settings:

```yaml
cases:
  - name: pglib_opf_case30_ieee
    file: pglib_opf_case30_ieee.m

  - name: pglib_opf_case2000_goc
    file: pglib_opf_case2000_goc.m
    ac:
      distributed_slack: false    # pandapower realloc crash
      pf_init: flat
    timeout: 2000                 # 15 min for large network

  - name: pglib_opf_case1354_pegase
    file: pglib_opf_case1354_pegase.m
    headroom_factor: 0.95         # tight topology
    ac:
      distributed_slack: false
      pf_init: flat
```

Shared compute parameters apply to every case unless overridden:

```yaml
compute:
  base_dispatch: ac_fpf
  dc:
    mode: materialize
    chunk_size: 64
    dtype: float64
    inj_std_mw: 10.0
  ac:
    chunk_size: 64
    balance: true
    pf_init: dc
    pf_solver: pandapower
    lossless: true
    distributed_slack: true
    trafo_model: pi

allow_download: true
data_dir: data/input
output_dir: run_artifacts/run_pglib_sweep/test_run_6
case_timeout_sec: 1200
```

### Sigma-radius experiments

Sigma-radius experiment configs (e.g., `sigma_case2000_goc.yaml`)
define:

```yaml
case:
  name: case2000_goc
  matpower_file: pglib_opf_case2000_goc.m
  slack_bus: 0

sigma_source: synthetic          # "synthetic" or "uc_jl"

synthetic_sigma:
  fraction: 0.10                 # sigma_P = 10% of |P_load| per bus
  power_factor: 0.9              # for sigma_Q estimation on zero-Q-load buses

verification:
  top_k: 5                       # number of tightest lines to verify
  scales: [0.5, 1.0, 1.5]        # perturbation scale factors

monte_carlo:
  enabled: true
  n_samples: 2000
  seed: 42
  tightened_limit:
    enabled: true
    target_r_sigma: 2.0
    n_samples: 2000

plot:
  top_k_critical: 5
  figsize: [20, 16]
  dpi: 200
```

### UnitCommitment.jl experiments

Multi-hour experiments using UC.jl time-series data:

```yaml
uc_jl:
  case_name: case118
  date: "2017-01-01"
  dest_dir: data/uc_jl
  power_factor: 0.9

verification:
  top_k: 10
  scales: [0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5]
```

---

## Full example configs

### Minimal compute run (IEEE 30-bus, DC only)

```yaml
extends: config_shared.yaml

compute:
  input: data/input/pglib_opf_case30_ieee.m
  slack_bus: 0
  base_dispatch: case
  dc:
    compute: true
  ac:
    compute: false
```

### Full DC+AC run with OPF dispatch

```yaml
extends:
  - ./config_shared.yaml
  - ./config_compute.yaml

compute:
  input: data/input/pglib_opf_case118_ieee.m
  slack_bus: 0
  base_dispatch: dc_opf
  dc:
    compute: true
    mode: materialize
    chunk_size: 64
    inj_std_mw: 10.0
  ac:
    compute: true
    chunk_size: 64
    balance: true
    pf_init: dc
  output:
    export_results: results/case118.json

opf:
  headroom_factor: 0.95
  threads: 4
```

### DC extensions (probabilistic + N-1)

```yaml
extends:
  - ../config.yaml
  - ../config_dc_extensions.yaml

compute:
  input: data/input/pglib_opf_case30_ieee.m
  dc:
    mode: materialize            # required for N-1
    probabilistic:
      enabled: true
    nminus1:
      enabled: true
      update_sensitivities: true
      islanding: skip
```

### AC sigma-radius with uniform sigma

```yaml
extends: ../config.yaml

compute:
  input: data/input/pglib_opf_case30_ieee.m
  base_dispatch: ac_fpf
  ac:
    compute: true
    sigma:
      sigma_p_mw_source: uniform
      sigma_q_mvar_source: uniform
      sigma_p_mw_uniform: 5.0
      sigma_q_mvar_uniform: 2.0
    metric:
      enabled: true
    save_h_vectors: true
```

### Monte Carlo verification

```yaml
extends: ../config.yaml

monte_carlo:
  mode: ac
  results: verification/results/case30.json
  input: data/input/pglib_opf_case30_ieee.m
  slack_bus: 0
  sampling:
    n_samples: 100000
    seed: 42
    chunk_size: 512
  tolerances:
    feas_tol: 0.0
    cert_tol: 1.0
  ac:
    sigma_p_mw: 5.0
    sigma_q_mvar: 2.0
```

---

## Performance-critical parameters

The following parameters have the largest impact on runtime and memory.
Tune them based on your system and network size.

### Memory

| Parameter | Impact | Guidance |
|-----------|--------|----------|
| `dc.mode` | `"materialize"` builds a dense (n_lines x n_buses) matrix. For a 10000-bus network with 15000 lines this is ~1.1 GB in float64. | Use `"operator"` unless you need N-1 analysis. |
| `dc.dtype` | `"float32"` halves memory vs `"float64"`. | Use `"float64"` for research; `"float32"` for exploratory runs on memory-constrained systems. |
| `dc.chunk_size` / `ac.chunk_size` | Controls batch size for LU-solve operations. Larger chunks use more peak memory but amortize Python overhead. | Start with 256. Reduce to 64 for very large networks (>5000 buses). |
| `monte_carlo.sampling.n_samples` | Each sample requires a full PF evaluation (AC mode). 50000 AC PF runs on a 2000-bus network can take hours. | Use 2000--5000 for quick checks, 50000+ for paper-quality results. |

### Runtime

| Parameter | Impact | Guidance |
|-----------|--------|----------|
| `compute.base_dispatch` | `"ac_fpf"` solves a nonlinear interior-point OPF, which can be 10--100x slower than `"case"` or `"dc_opf"`. | Use `"case"` for quick testing; `"dc_opf"` for dispatch optimization; `"ac_fpf"` for full AC feasibility. |
| `ac_fpf.max_attempts` | Each attempt runs a full PDIPM solve. With 3 attempts and 240s timeout, worst case is ~12 minutes. | Use 1 for fast runs; 3 for robust convergence on hard networks. |
| `ac_fpf.per_attempt_timeout` | Prevents a single slow PDIPM solve from blocking the entire pipeline. | Set to 120--240s for production runs. |
| `ac_fpf.max_iteration` | More iterations mean longer solves but better convergence. | 300 is a good default. Increase to 500 for challenging networks (PEGASE). |
| `opf.threads` | More threads can speed up the LP but may change the solution due to non-deterministic parallel pivoting. | Default `4` is a speed/consistency trade-off; use `1` for the strictest reproducibility. |

### Convergence

| Parameter | Impact | Guidance |
|-----------|--------|----------|
| `opf.headroom_factor` | Values too close to 1.0 leave no margin and may produce zero-radius results. Values too low (<0.90) can make the OPF infeasible. | 0.95--0.98 is the practical range. |
| `ac_fpf.vm_min_pu` / `vm_max_pu` | Tight bounds ([0.95, 1.05]) improve solution quality but can cause PDIPM infeasibility. Wide bounds ([0.8, 1.2]) improve convergence. | Start with [0.9, 1.1]; widen on convergence failure. |
| `ac_fpf.max_loading_percent` | Must be below 100 to compensate for PDIPM tolerance. Too low wastes capacity. | 99.0 is a safe default. Use 95.0 for PEGASE networks with known tolerance issues. |
| `ac.pf_init` | `"dc"` provides better starting points on most networks. `"flat"` is safer for networks with many transformers or unusual topology. | Use `"dc"` by default; switch to `"flat"` if DC init causes divergence. |

---

## Reproducibility contract

The project guarantees reproducible results across programmatic and
CLI invocations **if and only if** the following parameters match
between the Python defaults and the YAML defaults:

| Python constant / field | YAML key | Required value |
|------------------------|----------|----------------|
| `DEFAULT_UNCONSTRAINED_LINE_NOM_MW` | `opf.unconstrained_line_nom_mw` | `1e5` |
| `DEFAULT_OPF.headroom_factor` | `opf.headroom_factor` | `0.98` |
| `DEFAULT_MC.seed` | `monte_carlo.sampling.seed` / `report.sampling.seed` | `42` |
| `DEFAULT_MC.n_samples` | `monte_carlo.sampling.n_samples` / `report.sampling.n_samples` | `50000` |

If these values diverge between the code and YAML, CI tests and CLI
experiments will silently use different parameters.

### Run artifact tracking

Every CLI run saves the resolved configuration to the run directory:

| File | Format | Contents |
|------|--------|----------|
| `argv.txt` | Text | The exact command-line invocation |
| `config_source.yaml` | YAML | Copy of the source YAML config file |
| `config.json` | JSON | Fully resolved configuration (after CLI override) |
| `config.yaml` | YAML | Same as above, in YAML format |

These files enable exact reproduction of any past run.
