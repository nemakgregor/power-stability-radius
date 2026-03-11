# Limitations and Assumptions

This document catalogs all visible limitations and assumptions in the project, separated into confirmed (visible in code) and inferred (based on architecture choices).

> Cross-references: [mathematical_foundations.md](mathematical_foundations.md) for the formulations, [scientific_concepts.md](scientific_concepts.md) for the research context.

---

## 1. Mathematical Assumptions

### 1.1 Linearity of Sensitivity (Confirmed)

**Source**: `radii/core_l2.py`, `radii/ac_l2.py`, `radii/ac_sigma_radius.py`

All radius computations rely on a **first-order linear approximation**:
- DC: Δf = H · Δp (exact for lossless DC model)
- AC: Δ|S| ≈ h^T · Δu (first-order Taylor expansion around the base AC PF point)

**Implication**: The certificate is only valid within the region where the linearization is accurate. For large perturbations (large radius values), the actual nonlinear flow deviation may differ significantly from the linear prediction. The project explicitly notes this in comments as a "linearized magnitude model."

### 1.2 Gaussian Perturbation Model (Confirmed)

**Source**: `radii/probabilistic.py`, `radii/ac_sigma_radius.py`

The sigma-radius and overload probability computations assume:
- Bus injection perturbations are Gaussian: ΔP ~ N(0, σ_p²), ΔQ ~ N(0, σ_q²)
- Perturbations are independent across buses (diagonal covariance)
- The linearized flow perturbation inherits a Gaussian distribution

**Limitation**: Real-world perturbations (renewable intermittency, demand fluctuations) are typically:
- Non-Gaussian (heavy-tailed, bimodal)
- Correlated across buses (spatial correlation of wind/solar)
- Bounded (generation/load have physical limits)

### 1.3 Symmetric Line Limits (Confirmed)

**Source**: `radii/core_l2.py` docstring: "symmetric line limits: |f| ≤ c"

All formulations assume **symmetric** thermal limits: flow is bounded by the same limit in both directions. This is standard for thermal limits but does not capture:
- Directional stability limits
- Voltage collapse limits
- Transient stability constraints

### 1.4 Lossless Model for AC Certificate (Confirmed)

**Source**: `base_point/pandapower_tools.py: apply_lossless_policy_to_pandapower_net()`

The AC certificate deliberately uses a **lossless, series-only model**:
- Line resistance r = 0
- Shunt capacitance c = 0
- Shunt conductance g = 0
- Shunt elements (shunt, ward, xward) disabled
- Transformer resistive losses vkr_percent = 0

**Rationale** (from code docstring): "This aligns AC PF / AC MC with the certificate's internal linearization, which uses a series-only Ybus model (no shunt elements). Without removing shunt elements, the verification PF includes voltage-dependent admittances that the Jacobian does not model, causing systematic line-flow prediction errors."

**Limitation**: The lossless model introduces error relative to the true lossy network. The radius certificate is valid only for the lossless model, not for the full lossy network.

### 1.5 Power Balance Constraint (Confirmed)

**Source**: `radii/core_l2.py` docstring on balanced disturbances, `radii/ac_sigma_radius.py` balance logic

The balanced disturbance assumption requires ΣΔP = 0 (and ΣΔQ = 0 for AC). This is physically realistic (total generation must equal total load plus losses), but assumes:
- A single slack bus absorbs all imbalance
- No generator ramping limits constrain the balance
- The balance subspace projection correctly models the feasible perturbation set

### 1.6 Static Thermal Limits Only (Inferred)

The project only considers **static thermal ratings** (rateA from MATPOWER). It does not model:
- Dynamic thermal ratings (temperature-dependent)
- Short-term emergency ratings (rateB, rateC)
- Seasonal rating variations

---

## 2. Implementation Assumptions

### 2.1 Pandapower as Intermediate Representation (Confirmed)

**Source**: All parsers output pandapower networks, all models consume them

The entire pipeline assumes:
- Input can be converted to a pandapower network
- pandapower's internal conventions (bus indexing, per-unit system, element tables) are correct
- pandapower's AC PF solver is numerically reliable

**Limitation**: Networks that exercise pandapower edge cases (e.g., 3-winding transformers, DC links, storage elements) may not be fully supported.

### 2.2 Single Connected Component (Confirmed)

**Source**: `dc/dc_model.py: _check_connected_to_slack()`

The DC operator explicitly checks that all buses are connected to the slack bus via a union-find algorithm. If the network has isolated components, a `ValueError` is raised.

**Limitation**: Multi-island networks are not supported. Networks with intentional islanding (e.g., N-1 analysis where an outage causes islanding) require special handling via the `nminus1_islanding` parameter.

### 2.3 Sorted Bus/Line Ordering (Confirmed)

**Source**: Throughout `dc_model.py`, `ac_model.py`, `workflows.py`

All arrays are indexed according to `sorted(net.bus.index)` and `sorted(net.line.index)`. This is a design choice for determinism but assumes:
- Bus and line indices are integer-like and sortable
- The sorted order is stable across runs

### 2.4 Single ext_grid (Soft Assumption)

**Source**: `base_point/pandapower_tools.py: resolve_slack_bus_id()`

While the code handles multiple ext_grids with a warning, it fundamentally assumes **one slack bus**:
- The reduced system eliminates exactly one bus
- The Jacobian has n_bus - 1 equations
- Multiple ext_grids are disambiguated by choosing the smallest in-service ext_grid bus id

### 2.5 PQ and PV Bus Classification (Confirmed - AC Only)

**Source**: `ac/ac_model.py: _detect_pv_buses()`

The AC model classifies buses as:
- **Slack**: One slack bus, eliminated from the system
- **PV**: Generator buses with voltage control (V fixed, P specified, Q free)
- **PQ**: Load buses (both P and Q are specified)

**Limitation**: This classification is determined from `net.gen` and `net.ext_grid` at model build time. Generators that hit reactive power limits (switching from PV to PQ) are not modeled — the Jacobian is fixed at the base point.

---

## 3. Scalability Limitations

### 3.1 Dense H Matrix (DC Only)

**Source**: `dc/dc_model.py: materialize_H_full()`

The DC L2 radius computation materializes the full PTDF matrix H (m_lines × n_buses) as a dense numpy array. For large networks:
- case10000: H is 13193 × 10000 = ~1 GB at float64
- Memory scales as O(m × n)

The DCOperator supports matrix-free computation via `row_sensitivities_transposed()` (chunked LU solves), which is used for metric/sigma radii.

### 3.2 Sparse LU Factorization (Both DC and AC)

**Source**: `scipy.sparse.linalg.splu` in both models

LU factorization is the computational bottleneck. Scaling:
- Typical power networks (planar-like graphs): O(n^{1.5})
- Each adjoint solve: O(n) (triangular back-substitution)
- Total for m lines: O(n^{1.5} + m × n)

For very large networks (>10000 buses), the LU factorization time dominates.

### 3.3 AC Chunked Adjoint Solves

**Source**: `radii/ac_l2.py`, configurable via `ac_chunk_size`

AC L2 radius processes lines in chunks (default 256). Each chunk requires:
- Constructing the adjoint RHS matrix B (n_vars × chunk_size)
- Solving J^T Y = B (using the pre-factored LU)
- Memory: O(n_vars × chunk_size) per chunk

This bounds peak memory but doesn't reduce total computation.

### 3.4 Monte Carlo Scaling

**Source**: `verification/monte_carlo.py`

Monte Carlo verification requires n_samples × (one flow evaluation) per experiment:
- DC mode: Very fast (matrix multiplication)
- AC mode: Requires a full AC PF per sample (expensive for large networks)

For AC mode with 10000 samples on a 2000-bus network, this can take minutes to hours.

---

## 4. Solver-Related Limitations

### 4.1 HiGHS Solver (DC OPF)

- HiGHS is used via PyPSA/linopy for DC OPF
- The quality of the OPF solution depends on HiGHS settings (tolerance, time limit)
- HiGHS may produce slightly different solutions across versions

### 4.2 pandapower.runpp (AC PF)

- Uses Newton-Raphson power flow
- May not converge for heavily loaded or ill-conditioned networks
- The 3-attempt retry cascade mitigates this but is not guaranteed to succeed

### 4.3 pandapower.runopp (AC FPF)

- Uses PIPS (Primal Interior Point Solver) internally
- Convergence is sensitive to initial point and tolerance settings
- Phase-shifting transformers can cause numerical difficulties

---

## 5. Numerical Issues

### 5.1 Near-Zero Sensitivities

**Source**: All radius modules use `eps_norm` thresholds (typically 1e-12)

When ||g_l|| ≈ 0 (a line has near-zero sensitivity to all bus injections), the radius becomes numerically infinite. The code handles this by returning `float("inf")` when margin ≥ 0.

### 5.2 Near-Zero Apparent Power

**Source**: `radii/ac_l2.py`, `_FALLBACK_WP_WQ = 1/sqrt(2)`

When |S0| ≈ 0 at a line end, the gradient of the norm ||S|| is undefined. The code falls back to equal P/Q weights (w_P = w_Q = 1/√2), which is conservative but may not be tight.

### 5.3 Voltage Angle Unit Checking

**Source**: `ac/ac_model.py: _build_reduced_pf_jacobian_mw_per_unit()`

The code validates that voltage angles are in radians (max|va| < 10). This guards against silent unit mismatches (degrees vs radians), which would produce completely wrong Jacobians.

### 5.4 Negative Reactance

**Source**: `dc/dc_model.py` contains explicit handling and warning for lines with negative series reactance (b < 0), which can occur with series compensation or capacitive modeling.

---

## 6. Cases Not Supported

| Feature | Status | Notes |
|---------|--------|-------|
| Three-winding transformers | Not supported | Only two-winding trafos modeled |
| HVDC links | Not supported | DC elements not parsed |
| Storage elements | Not supported | Not included in dispatch model |
| Switched shunts | Partially | Disabled under lossless policy |
| Generator reactive limits | Not modeled | PV buses stay PV (no Q-limit switching) |
| Multiple slack buses | Partial | Uses smallest in-service ext_grid bus id |
| Contingency analysis beyond N-1 | Not implemented | Only single-line outages |
| Time-series analysis | Not implemented | Operates on a single snapshot |
| Voltage stability | Not addressed | Only thermal limits considered |

---

## 7. Possible Failure Modes

### 7.1 Silent Accuracy Degradation

The linearized certificate becomes less accurate as the actual perturbation magnitude grows. There is no automatic detection of when the linear approximation breaks down. The Monte Carlo verification serves as a post-hoc check.

### 7.2 Overly Optimistic Radius

If the base point is obtained from a dispatch (e.g., DC OPF) that doesn't fully capture network physics, the AC PF base point may show different loading patterns, potentially making the radius non-conservative.

### 7.3 Parser Failures

The MATPOWER parser has a primary path (pandapower's `from_mpc()`) and a fallback regex parser. Some non-standard .m files may fail both parsing strategies.

---

## 8. Technical Debt

### Identified from code inspection:

1. **Large monolithic `workflows.py`** (~1400 lines): The main pipeline function is very long. Helper functions are extracted but the control flow is complex.

2. **Redundant base quantity computation**: `LineBaseQuantities` is computed multiple times when both L2 and sigma/metric radii are requested (mitigated by passing `base` parameter).

3. **Import-time optional dependencies**: Several modules use try/except for scipy imports. This is correct but means missing scipy is only detected at runtime.

4. **AC model assumes series-only topology**: The Ybus construction explicitly excludes shunt elements, which is intentional but limits the model's applicability to networks where shunt elements are significant.
