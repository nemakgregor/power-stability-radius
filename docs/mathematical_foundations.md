# Mathematical Foundations of Power Stability Radius

This document provides a complete and self-contained mathematical reference for
every formulation implemented in the `power-stability-radius` codebase.  All
notation is defined before first use and is consistent across sections.

---

## Table of Contents

1. [Problem Statement and Notation](#1-problem-statement-and-notation)
2. [DC Power Flow Model](#2-dc-power-flow-model)
3. [L2 Stability Radius Certificate (DC)](#3-l2-stability-radius-certificate-dc)
4. [Balanced Disturbance Projection](#4-balanced-disturbance-projection)
5. [AC Power Flow Model](#5-ac-power-flow-model)
6. [AC L2 Stability Radius](#6-ac-l2-stability-radius)
7. [Sigma-Radius: Probabilistic Certificate (DC)](#7-sigma-radius-probabilistic-certificate-dc)
8. [AC Sigma-Radius](#8-ac-sigma-radius)
9. [Metric Radius (DC)](#9-metric-radius-dc)
10. [AC Metric Radius](#10-ac-metric-radius)
11. [N-1 Contingency Radius](#11-n-1-contingency-radius)
12. [Baseline Metrics](#12-baseline-metrics)
13. [Comparison of Formulations](#13-comparison-of-formulations)
14. [Summary Comparison Table](#14-summary-comparison-table)
15. [Implementation File Map](#15-implementation-file-map)

---

## 1. Problem Statement and Notation

### 1.1 Problem Statement

Consider a power network operating at a feasible base point.  External
disturbances -- fluctuations in renewable generation, load variations, or
forecast errors -- perturb bus injections away from their scheduled values.  The
**stability radius** quantifies the largest perturbation magnitude (in a chosen
norm) that the network can absorb without violating any thermal line limit.

Formally, let `f(p)` map bus injections `p` to line flows.  Given a base
operating point `p_0` with base flows `f_0 = f(p_0)` and symmetric thermal
limits `c`, the stability radius is:

```
r* = max { r >= 0 : ||Delta_p|| <= r  ==>  |f_0 + Delta_f| <= c  componentwise }
```

where `Delta_f` is the flow change induced by `Delta_p` and the norm `||.||`
is chosen to reflect the disturbance model.

### 1.2 Sets and Indices

| Symbol | Description |
|--------|-------------|
| `N` | Set of buses, `|N| = n` |
| `L` | Set of monitored lines (branches), `|L| = m` |
| `N_s` | Slack bus (reference, one element) |
| `N_PV` | Set of PV (voltage-controlled / generator) buses |
| `N_PQ` | Set of PQ (load) buses; `N_PQ = N \ (N_s union N_PV)` |
| `i, k` | Bus indices |
| `l` | Line (branch) index |
| `n_red = n - 1` | Dimension of the reduced system (slack eliminated) |
| `n_pq = |N_PQ|` | Number of PQ buses |

### 1.3 Parameters

| Symbol | Units | Description |
|--------|-------|-------------|
| `V_i` | kV (nominal) or pu (operating) | Bus voltage magnitude |
| `theta_i` | rad | Bus voltage angle |
| `X_l` | Ohm | Series reactance of branch `l` |
| `R_l` | Ohm | Series resistance of branch `l` (set to 0 in lossless mode) |
| `b_l` | MW/rad | DC branch susceptance coefficient |
| `c_l` | MW (DC) or MVA (AC) | Symmetric thermal limit for line `l` |
| `S_base` | MVA | System base apparent power (`sn_mva`) |
| `Z_base` | Ohm | Base impedance: `V_kV^2 / S_base` |
| `sigma_P_i` | MW | Standard deviation of active power injection at bus `i` |
| `sigma_Q_i` | MVAr | Standard deviation of reactive power injection at bus `i` |

### 1.4 Variables and Derived Quantities

| Symbol | Description |
|--------|-------------|
| `p` | Vector of net bus active power injections (MW) |
| `q` | Vector of net bus reactive power injections (MVAr) |
| `f_l` | Active power flow on line `l` (MW, DC model) |
| `S_l` | Apparent power flow on line `l` (MVA, AC model) |
| `Delta_p` | Perturbation of bus injections |
| `Delta_f` | Induced perturbation of line flows |
| `m_l` | Thermal margin for line `l`: `m_l = c_l - |f_0_l|` (DC) or `c_l - |S_0_l|` (AC) |
| `H` | Sensitivity (PTDF) matrix: `Delta_f = H * Delta_p` |
| `g_l` | Row `l` of `H` (sensitivity of flow `l` to injections) |
| `J` | AC power flow Jacobian |
| `h_l` | Adjoint sensitivity vector for line `l` (AC model) |

---

## 2. DC Power Flow Model

**Implementation:** `src/stability_radius/dc/dc_model.py`

### 2.1 Assumptions

The DC power flow approximation assumes:

1. Lossless transmission lines (`R = 0`)
2. Small voltage angle differences across branches (`sin(theta) approx theta`)
3. Flat voltage profile (`V_i approx 1.0` pu for all buses)

Under these assumptions, active power flow is linear in voltage angles.

### 2.2 Branch Susceptance Coefficient

For each branch `l` connecting buses `i` (from) and `k` (to):

```
b_l = V_kV^2 / X_ohm       [MW/rad]
```

where `V_kV` is the nominal voltage at the from-bus and `X_ohm` is the total
series reactance.  For a transmission line:

```
X_ohm = x_ohm_per_km * length_km / parallel
```

For a transformer, the reactance is derived from short-circuit data:

```
z_pu = vk_percent / 100
r_pu = vkr_percent / 100
x_pu = sqrt(max(z_pu^2 - r_pu^2, 0))
Z_base = V_hv_kV^2 / S_trafo_MVA
X_ohm = x_pu * Z_base
```

### 2.3 Transformer Tap Modeling (DC)

Transformer tap ratios modify the effective susceptance:

```
tap = 1 + (tap_pos - tap_neutral) * tap_step_percent / 100
b_eff = b_raw / tap
```

If `tap_side = "lv"`, the ratio is inverted: `tap <- 1/tap`.

### 2.4 Oriented Incidence Matrix

Let `A` be the oriented incidence matrix of dimension `(m_branches x n)` where
`m_branches` includes all branch types (lines, transformers, impedances) used
to build the nodal susceptance matrix.  For each branch `l` from bus `i` to
bus `k`:

```
A[l, i] = +1,   A[l, k] = -1,   all other entries zero
```

### 2.5 Nodal Susceptance Matrix

```
B = A^T * diag(b) * A
```

where `b` is the vector of all branch susceptance coefficients.  `B` is
`(n x n)`, symmetric, and singular (rank `n-1`).

### 2.6 DC Power Flow Equations

Line flows:

```
f_l = b_l * (theta_i - theta_k)
```

In matrix form:

```
f = diag(b) * A * theta
```

Nodal power balance:

```
B * theta = p
```

### 2.7 Phase Shifter Support

When transformers have a nonzero `shift_degree` (phase shifting transformers),
the absolute power flow equation becomes:

```
B_red * theta_red = p_red + shift_inj_red
```

where the constant right-hand-side correction is:

```
shift_inj_red = A_red^T * diag(b) * shift_rad
```

Here `shift_rad` is the vector of phase shifts in radians and `A_red` is the
incidence matrix with the slack bus column removed.  The shift term drops out of
perturbation equations because it is a constant.

### 2.8 Reduced System

The slack bus (reference bus with `theta_s = 0`) is eliminated.  Define
`mask_non_slack` as the boolean mask excluding the slack position:

```
B_red = B[mask, :][:, mask]        (n-1) x (n-1), invertible
p_red = p[mask]                    (n-1)
theta_red = B_red^{-1} * (p_red + shift_inj_red)
```

In practice, `B_red` is factored once (sparse LU via SciPy) and reused for all
subsequent solves.

### 2.9 PTDF (Sensitivity) Matrix

The Power Transfer Distribution Factor matrix `H` maps injection perturbations
to line flow changes.  For monitored lines only:

```
W = diag(b_lines) * A_lines_red       (m x (n-1))
H_red = W * B_red^{-1}                (m x (n-1))
```

Extending to full bus dimension (slack column is zero):

```
H_full[l, i] = H_red[l, red_pos(i)]   for i != slack
H_full[l, slack] = 0
```

The key sensitivity relation is:

```
Delta_f = H * Delta_p
```

This is exact under the DC model: flows respond linearly to injection changes.

---

## 3. L2 Stability Radius Certificate (DC)

**Implementation:** `src/stability_radius/radii/core_l2.py`

### 3.1 Setup

Given:
- Linear flow map: `Delta_f = H * Delta_p`
- Base flows: `f_0 in R^m`
- Symmetric line limits: `|f_l| <= c_l` for all `l`

### 3.2 Per-Line Margin

```
m_l = c_l - |f_0_l|
```

The margin is positive when the base point is feasible for line `l`.

### 3.3 Per-Line Sensitivity Row

```
g_l = H[l, :]     in R^n
```

### 3.4 Per-Line L2 Radius

```
r_l = m_l / ||g_l||_2
```

### 3.5 Global Certificate

```
r* = min_{l in L} r_l
```

### 3.6 Certificate Guarantee (Cauchy-Schwarz)

**Theorem.** If the base point is feasible (`m_l >= 0` for all `l`), then for
any perturbation `Delta_p` with `||Delta_p||_2 <= r*`, all line constraints
remain satisfied:

```
|f_0_l + g_l^T * Delta_p| <= c_l     for all l in L
```

**Proof.**  By the Cauchy-Schwarz inequality:

```
|g_l^T * Delta_p| <= ||g_l||_2 * ||Delta_p||_2
                   <= ||g_l||_2 * r*
                   <= ||g_l||_2 * (m_l / ||g_l||_2)
                   = m_l
```

Therefore:

```
|f_0_l + g_l^T * Delta_p| <= |f_0_l| + |g_l^T * Delta_p|
                           <= |f_0_l| + m_l
                           = |f_0_l| + (c_l - |f_0_l|)
                           = c_l
```

### 3.7 Edge Cases

- If `||g_l||_2 = 0` and `m_l >= 0`: line `l` is insensitive to injections;
  `r_l = +inf`.
- If `||g_l||_2 = 0` and `m_l < 0`: base point already infeasible;
  `r_l = -inf`.
- If no finite radii exist: `r* = NaN` (degenerate system).

---

## 4. Balanced Disturbance Projection

**Implementation:** `src/stability_radius/radii/core_l2.py` (helper functions)

### 4.1 Motivation

In real power systems, injections are constrained by power balance: the total
generation must equal total load plus losses.  Under the DC (lossless)
approximation this requires:

```
1^T * Delta_p = 0
```

That is, the perturbation must lie in the sum-zero hyperplane.

### 4.2 Dual Norm on the Balanced Subspace

Restricting `Delta_p` to the subspace `{x : 1^T x = 0}` modifies the dual norm
of the sensitivity vector.  The worst-case flow change per unit of balanced
perturbation is:

```
max_{||Delta_p||_2 <= 1, 1^T Delta_p = 0}  g_l^T * Delta_p  =  ||Proj(g_l)||_2
```

where `Proj` is the orthogonal projection onto the sum-zero subspace:

```
Proj(g) = g - mean(g) * 1
```

### 4.3 Closed-Form Computation

```
||Proj(g)||_2^2 = ||g||_2^2 - (sum(g))^2 / n
```

This is numerically stable and avoids explicitly forming the projection matrix.

### 4.4 Balanced L2 Radius

```
r_l^{bal} = m_l / ||Proj(g_l)||_2
```

### 4.5 Slack Invariance

The balanced radius is invariant to the choice of slack bus.  Since the PTDF
row `g_l` depends on which bus is chosen as slack (the column of `H` for the
slack bus is zero), a change of slack reference shifts `g_l` by a constant
vector `alpha * 1`.  The projection `Proj(g_l)` removes this constant
component, so the balanced radius is independent of slack choice.

---

## 5. AC Power Flow Model

**Implementation:** `src/stability_radius/ac/ac_model.py`

### 5.1 Bus Admittance Matrix (Ybus)

The AC model uses a series-only admittance representation (no shunt elements).

**Lines:** For line `l` from bus `i` to bus `k`:

```
z_ohm = (r_ohm_per_km + j * x_ohm_per_km) * length_km / parallel
z_pu  = z_ohm / Z_base,     where Z_base = V_kV^2 / S_base
y_l   = 1 / z_pu
```

Under the project's lossless policy (`lossless=True`), `r = 0` for all
branches, so `z_pu = j * x_pu` and `y_l` is purely imaginary.

Ybus stamping for a simple series branch:

```
Y_ii += y_l
Y_kk += y_l
Y_ik -= y_l
Y_ki -= y_l
```

**Transformers with complex tap ratio:**  Let `a = tap * e^{j*phi}` be the
complex tap ratio on the HV side.  The Ybus stamping with tap is:

```
Y_ii += y / |a|^2
Y_kk += y
Y_ik -= y / conj(a)
Y_ki -= y / a
```

where `i` is the HV bus, `k` is the LV bus, and `y` is the series admittance
in per-unit.

### 5.2 PV/PQ Bus Classification

Buses are classified as:

- **Slack** (`N_s`): fixed `V` and `theta`, excluded from the reduced system.
- **PV** (`N_PV`): fixed `V` magnitude (generator with voltage control). Only
  `theta` is a free variable; the Q equation is excluded.
- **PQ** (`N_PQ`): both `theta` and `V` are free variables; both P and Q
  equations are included.

### 5.3 Reduced Power Flow Jacobian

The Jacobian linearizes the AC power flow equations around the base point.  The
reduced system has:

- **Variables:** `x = [theta_non_slack; V_pq]` with dimension `n_theta + n_pq`
  where `n_theta = n - 1`.
- **Equations:** `[dP_non_slack; dQ_pq]` with the same dimension.

The Jacobian has a 4-block structure:

```
J = [ dP/d_theta    dP/dV_pq  ]    rows: P equations (n_theta)
    [ dQ/d_theta    dQ/dV_pq  ]    rows: Q equations (n_pq)
```

### 5.4 Jacobian Entries

**Off-diagonal entries** (for buses `i != k` connected by admittance `Y_ik = G_ik + j*B_ik`):

Let `theta_ik = theta_i - theta_k`.

```
dP_i/d_theta_k = V_i * V_k * (G_ik * sin(theta_ik) - B_ik * cos(theta_ik)) * S_base

dP_i/dV_k      = V_i * (G_ik * cos(theta_ik) + B_ik * sin(theta_ik)) * S_base

dQ_i/d_theta_k = -V_i * V_k * (G_ik * cos(theta_ik) + B_ik * sin(theta_ik)) * S_base

dQ_i/dV_k      = V_i * (G_ik * sin(theta_ik) - B_ik * cos(theta_ik)) * S_base
```

**Diagonal entries** (with `Y_ii = G_ii + j*B_ii`):

```
dP_i/d_theta_i = (-Q_i - B_ii * V_i^2) * S_base

dP_i/dV_i      = (P_i / V_i + G_ii * V_i) * S_base

dQ_i/d_theta_i = (P_i - G_ii * V_i^2) * S_base

dQ_i/dV_i      = (Q_i / V_i - B_ii * V_i) * S_base
```

Here `P_i` and `Q_i` are the per-unit net injections computed from the base
point: `S_i = V_i * conj(I_i)`, `I = Y_bus * V`.

The Jacobian is factored (sparse LU) once and reused for all adjoint solves.

### 5.5 Adjoint Solve

The first-order relation is:

```
J * dx = du
```

where `du = [dP_non_slack; dQ_pq]` are injection perturbations (MW, MVAr).
For sensitivity analysis, we need the adjoint:

```
a = J^{-T} * b
```

This maps a flow-side gradient `b` back to the injection space.

---

## 6. AC L2 Stability Radius

**Implementation:** `src/stability_radius/radii/ac_l2.py`

### 6.1 Apparent Power Flow

For each line `l` and each end (from/to), the apparent power magnitude is:

```
|S_0| = sqrt(P_0^2 + Q_0^2)
```

where `P_0` and `Q_0` are the active and reactive power flows at the
respective end in the base operating point.

### 6.2 Thermal Margin

```
m_l = c_l - |S_0_l|       [MVA]
```

where `c_l` is the thermal limit (MVA) and `|S_0_l|` is the apparent power at
the binding end.

### 6.3 Sensitivity of Apparent Power Magnitude

The apparent power magnitude `|S| = sqrt(P^2 + Q^2)` is a nonlinear function.
Its gradient with respect to `(P, Q)` is:

```
d|S|/dP = P / |S| = w_P
d|S|/dQ = Q / |S| = w_Q
```

**Fallback at zero flow:** When `|S_0| < epsilon` (near zero), the gradient of
the norm is undefined.  The implementation uses a conservative, unbiased
direction `(w_P, w_Q) = (1/sqrt(2), 1/sqrt(2))`.

### 6.4 Chain Rule: Flow-to-Injection Sensitivity

For a specific line `l` and end (from or to), let bus `i` be the near-end bus
and bus `k` the far-end bus.  The line's series admittance is `y = g + jb` (pu).
Define the angle difference `theta = theta_i - theta_k` and the auxiliary
quantities:

```
A = g * cos(theta) + b * sin(theta)
B_t = g * sin(theta) - b * cos(theta)
```

The per-unit flow derivatives at the `i`-side are:

```
dP/d_theta_i = V_i * V_k * B_t * S_base
dP/d_theta_k = -V_i * V_k * B_t * S_base
dQ/d_theta_i = -V_i * V_k * A * S_base
dQ/d_theta_k = V_i * V_k * A * S_base

dP/dV_i = (2*g*V_i - V_k*A) * S_base
dP/dV_k = -V_i * A * S_base
dQ/dV_i = (-2*b*V_i - V_k*B_t) * S_base
dQ/dV_k = -V_i * B_t * S_base
```

The composite right-hand side for the adjoint solve is:

```
b_entry = w_P * dP/dx + w_Q * dQ/dx
```

assembled into the appropriate positions of a vector `b in R^{n_vars}` (where
`n_vars = n_theta + n_pq`).

### 6.5 Adjoint Sensitivity Vector

Solving the adjoint system:

```
h_l = J^{-T} * b_l
```

The vector `h_l in R^{n_vars}` is partitioned as:

```
h_l = [h_l^P ; h_l^Q]
```

where `h_l^P in R^{n_theta}` and `h_l^Q in R^{n_pq}` are the sensitivities of
`|S_l|` to active and reactive power injections, respectively.

### 6.6 Per-Constraint L2 Radius

Without balance projection:

```
r_l = m_l / ||h_l||_2
```

With balanced disturbance projection (see Section 6.7):

```
r_l = m_l / ||h_l||_{bal}
```

### 6.7 Balanced Dual Norm (AC, Two-Block)

In the AC setting, the balanced disturbance constraint is:

```
1^T * Delta_P = 0     (active power balance)
1^T * Delta_Q = 0     (reactive power balance)
```

The adjoint vector `h_l` is partitioned into P-block and Q-block.  The
balanced dual norm projects each block independently:

```
Proj_P(h_l^P) = h_l^P - mean(h_l^P) * 1_n       (projection in R^{n_bus})
Proj_Q(h_l^Q) = h_l^Q - mean(h_l^Q) * 1_{n_pq}  (projection in R^{n_pq})
```

The combined balanced norm is:

```
||h_l||_{bal} = sqrt( ||Proj_P(h_l^P)||_2^2 + ||Proj_Q(h_l^Q)||_2^2 )
```

Each squared projected norm uses the closed form:

```
||Proj(v)||_2^2 = ||v||_2^2 - (sum(v))^2 / dim(v)
```

### 6.8 Binding End and Per-Line Radius

Both ends (from and to) of each line are evaluated.  The binding end is the one
with the smaller radius:

```
r_line = min(r_from, r_to)
binding_end = argmin(r_from, r_to)
```

### 6.9 Global AC L2 Certificate

```
r*_AC = min_{l in L} r_line_l
```

---

## 7. Sigma-Radius: Probabilistic Certificate (DC)

**Implementation:** `src/stability_radius/radii/probabilistic.py`

### 7.1 Stochastic Model

Injection perturbations are modeled as Gaussian:

```
Delta_p ~ N(0, Sigma)
```

where `Sigma` is the covariance matrix (or diagonal matrix of variances).

### 7.2 Flow Standard Deviation

Under the linear DC model `F = f_0 + g_l^T * Delta_p`, the flow on line `l` is
Gaussian with:

```
E[F_l] = f_0_l
Var(F_l) = g_l^T * Sigma * g_l
sigma_flow_l = sqrt(g_l^T * Sigma * g_l)
```

For diagonal `Sigma = diag(sigma_1^2, ..., sigma_n^2)`:

```
sigma_flow_l = sqrt( sum_i  sigma_i^2 * g_l_i^2 )
```

### 7.3 Sigma-Radius

The sigma-radius measures the thermal margin in units of flow standard
deviations:

```
r_sigma_l = m_l / sigma_flow_l = (c_l - |f_0_l|) / sigma_flow_l
```

This is dimensionless and answers: "How many standard deviations of flow
fluctuation fit within the available margin?"

### 7.4 Overload Probability (Gaussian)

For symmetric limits `|F| <= c` and base flow `f_0`:

```
P(|F| > c) = Q((c - |f_0|) / sigma_flow) + Q((c + |f_0|) / sigma_flow)
```

where `Q(x) = P(Z > x)` for `Z ~ N(0,1)` is the Gaussian Q-function,
implemented via `Q(x) = 0.5 * erfc(x / sqrt(2))` for numerical stability.

**Edge case:** If `sigma_flow = 0`, the random variable is degenerate:
`P(|F| > c) = 1` if `|f_0| > c`, else `0`.

---

## 8. AC Sigma-Radius

**Implementation:** `src/stability_radius/radii/ac_sigma_radius.py`

### 8.1 Setup

This formulation extends the sigma-radius to the AC model using precomputed
adjoint sensitivity vectors.

Let `h_l in R^{2n_bus}` be the adjoint sensitivity of `|S_l|` (at the binding
end) to injection perturbations `[Delta_P; Delta_Q]`, partitioned as:

```
h_l = [h_l^P ; h_l^Q]
```

Per-bus injection standard deviations: `sigma_P_i` (MW), `sigma_Q_i` (MVAr).

Diagonal covariance:

```
Sigma = diag(sigma_P_1^2, ..., sigma_P_n^2, sigma_Q_1^2, ..., sigma_Q_n^2)
```

### 8.2 Flow Standard Deviation (AC)

```
sigma_flow_l = ||Sigma^{1/2} * h_l||_2
             = sqrt( sum_i (sigma_P_i * h_l^P_i)^2 + sum_i (sigma_Q_i * h_l^Q_i)^2 )
```

### 8.3 AC Sigma-Radius

```
r_sigma_l = (c_l - |S_0_l|) / sigma_flow_l
```

### 8.4 Worst-Case Perturbation

The worst-case perturbation in physical units (MW/MVAr) that reaches the limit:

```
Delta_u_l* = r_sigma_l * (Sigma * h_l) / sigma_flow_l
```

Componentwise:

```
Delta_P_i* = r_sigma_l * sigma_P_i^2 * h_l^P_i / sigma_flow_l
Delta_Q_i* = r_sigma_l * sigma_Q_i^2 * h_l^Q_i / sigma_flow_l
```

This is the solution to the optimization problem:

```
max   h_l^T * Delta_u
s.t.  ||Sigma^{-1/2} * Delta_u||_2 <= r_sigma
```

The worst-case perturbation points in the `Sigma * h_l` direction, which
tilts towards buses with higher uncertainty (larger `sigma`).

### 8.5 Gaussian Overload Probability (AC)

```
P(|S| > c) = Q((c - |S_0|) / sigma_flow) + Q((c + |S_0|) / sigma_flow)
```

Same functional form as the DC case (Section 7.4), but using the AC-derived
`sigma_flow`.

### 8.6 Balanced Projection (Sigma-Weighted)

When the balance constraint `1^T Delta_P = 0` and `1^T Delta_Q = 0` is
enforced, the worst-case perturbation must satisfy these constraints.  Because
the perturbation has the form `Delta_P_i = r * sigma_P_i^2 * h_l^P_i / sigma_flow`,
the sum-zero constraint requires:

```
sum_i sigma_P_i^2 * h_l^P_i = 0
```

This is enforced by a **sigma-squared-weighted mean subtraction**:

```
h_l^P  <-  h_l^P - [sum(sigma_P^2 * h_l^P) / sum(sigma_P^2)]
h_l^Q  <-  h_l^Q - [sum(sigma_Q^2 * h_l^Q) / sum(sigma_Q^2)]
```

This differs from the L2 certificate (which uses unweighted mean subtraction)
because here the perturbation ellipsoid is anisotropic (sigma-weighted).

Formally, this is the Lagrangian solution to:

```
max   h^T * Delta_u
s.t.  ||Sigma^{-1/2} * Delta_u||_2 <= r
      1^T * Delta_P = 0
      1^T * Delta_Q = 0
```

---

## 9. Metric Radius (DC)

**Implementation:** `src/stability_radius/radii/metric.py`

### 9.1 Generalized Norm

For a symmetric positive definite (SPD) weight matrix `M`, the M-norm is:

```
||Delta_p||_M = sqrt(Delta_p^T * M * Delta_p)
```

This allows encoding problem-specific structure: for instance, setting
`M = diag(1/sigma^2)` recovers the sigma-radius; setting `M = I` recovers the
L2 radius.

### 9.2 Metric Radius Formula

For each line `l`:

```
r_l^{(M)} = m_l / sqrt(g_l^T * M^{-1} * g_l)
```

**Derivation.**  The dual norm of `g_l` under `||.||_M` is:

```
||g_l||_{M,*} = max_{||Delta_p||_M <= 1} g_l^T * Delta_p = sqrt(g_l^T * M^{-1} * g_l)
```

By the generalization of the Cauchy-Schwarz inequality to M-norms:

```
|g_l^T * Delta_p| <= ||g_l||_{M,*} * ||Delta_p||_M
```

Therefore, the metric radius `r_l^{(M)} = m_l / ||g_l||_{M,*}` certifies
feasibility for all `||Delta_p||_M <= r_l^{(M)}`.

### 9.3 Efficient Computation via Cholesky

For dense SPD `M`, the Cholesky factorization `M = L * L^T` allows:

```
g_l^T * M^{-1} * g_l = ||L^{-1} * g_l||_2^2
```

so:

```
sqrt(g_l^T * M^{-1} * g_l) = ||L^{-1} * g_l||_2
```

The Cholesky factor `L` is computed once and reused for all lines (forward
substitution `L * z = g_l` is `O(n^2)` per line).

### 9.4 Global Metric Certificate

```
r*_M = min_{l in L} r_l^{(M)}
```

---

## 10. AC Metric Radius

**Implementation:** `src/stability_radius/radii/ac_metric_radius.py`

### 10.1 Setup

This formulation extends the metric radius to the AC model using precomputed
adjoint sensitivity vectors `h_l in R^{2n_bus}`.

### 10.2 Formula

Given SPD weight matrix `M in R^{2n x 2n}` (or diagonal vector `m in R^{2n}`):

```
r_l^M = (c_l - |S_0_l|) / sqrt(h_l^T * M^{-1} * h_l)
```

### 10.3 Special Cases

- `M = I`:  `r_l^M = r_l^{L2}` (standard AC L2 radius).
- `M = Sigma^{-1}`:  `r_l^M = r_l^{sigma}` (AC sigma-radius), because
  `h^T * (Sigma^{-1})^{-1} * h = h^T * Sigma * h`.

### 10.4 Diagonal Optimization

When `M` is diagonal (as is typical: `M = diag(m_1, ..., m_{2n})`), the
computation avoids the `O(n^3)` Cholesky and uses:

```
h_l^T * M^{-1} * h_l = sum_i h_l_i^2 / m_i
```

in `O(n)` time per line.

### 10.5 Balanced Projection (AC Metric)

With balance, the P-block and Q-block of each `h_l` are projected via
**unweighted** mean subtraction (unlike the sigma-radius which uses
sigma-squared weighting):

```
h_l^P  <-  h_l^P - mean(h_l^P)
h_l^Q  <-  h_l^Q - mean(h_l^Q)
```

The metric norm is then computed on the projected vector.

---

## 11. N-1 Contingency Radius

**Implementation:** `src/stability_radius/radii/nminus1.py`

### 11.1 Concept

The N-1 contingency radius evaluates security under single-element outages.
For each contingency (outage of line `k`), the post-contingency flows and
sensitivities are approximated using Line Outage Distribution Factors (LODF).

### 11.2 PTDF for Line Transfers

The PTDF matrix for line endpoint transfers is:

```
PTDF_{m,k} = g_m^T * (e_{from(k)} - e_{to(k)})
```

In matrix form, with `E` the oriented incidence matrix:

```
PTDF = H * E^T         (m x m)
```

### 11.3 LODF Computation

The Line Outage Distribution Factor matrix is derived from the PTDF:

```
LODF_{m,k} = PTDF_{m,k} / (1 - PTDF_{k,k})     for m != k
LODF_{k,k} = -1
```

**Islanding detection:** When `1 - PTDF_{k,k} approx 0`, the outage of line `k`
would island part of the network.  The LODF is undefined and the column is set
to `NaN`.

### 11.4 Post-Contingency Quantities

For contingency `k` (outage of line `k`):

**Post-contingency flows:**

```
f^{(k)} = f_0 + LODF[:, k] * f_0_k
f_k^{(k)} = 0                          (outaged line carries zero flow)
```

**Post-contingency margins:**

```
m_l^{(k)} = max(c_l - |f_l^{(k)}|, 0)
```

**Post-contingency sensitivities** (optional Woodbury update):

```
g_m^{(k)} = g_m + LODF_{m,k} * g_k
```

### 11.5 Post-Contingency Balanced Norms

The balanced (projected) norm for the updated sensitivity is:

```
||Proj(g_m^{(k)})||_2^2 = ||g_m^{(k)}||_2^2 - (sum(g_m^{(k)}))^2 / n_bus
```

This is computed efficiently from precomputed quantities:

```
||g_m + alpha_m * g_k||^2 = ||g_m||^2 + 2*alpha_m*(g_m . g_k) + alpha_m^2 * ||g_k||^2

sum(g_m + alpha_m * g_k) = sum(g_m) + alpha_m * sum(g_k)
```

where `alpha_m = LODF_{m,k}`.

### 11.6 Effective N-1 L2 Radius

For each monitored line `m`, the N-1 radius is the worst case over all
contingencies:

```
r_m^{N-1} = min_{k != m}  m_m^{(k)} / ||Proj(g_m^{(k)})||_2
```

The worst contingency index `k*` is also reported for each monitored line.

---

## 12. Baseline Metrics

**Implementation:** `src/stability_radius/metrics/ac_baselines.py`

These are simple, widely-used heuristic metrics that serve as benchmarks against
which the stability-radius certificates can be compared.

### 12.1 Loading Ratio

```
LR_l = |S_0_l| / c_l
```

Range: `[0, inf)`.  Values approaching `1.0` indicate the line is near its
thermal limit.  Values exceeding `1.0` indicate overload.

### 12.2 Headroom

```
HR_l = c_l - |S_0_l|      [MVA]
```

The absolute margin before overload.  Can be negative if the line is already
overloaded.

### 12.3 Cantelli (Chebyshev) Upper Bound

A distribution-free upper bound on overload probability using only the first
two moments:

```
P(|S| > c) <= sigma^2 / (sigma^2 + m^2)
```

where `m = c - |S_0|` is the headroom and `sigma` is the flow standard
deviation.  This bound:

- Is valid for **any** distribution (not just Gaussian), requiring only finite
  mean and variance.
- Equals `1.0` when `m <= 0` (already overloaded or binding).
- Equals `0.0` when `sigma = 0` and `m > 0` (deterministic and feasible).

---

## 13. Comparison of Formulations

### 13.1 DC vs. AC Model

| Aspect | DC Model | AC Model |
|--------|----------|----------|
| Flow quantity | Active power `P` (MW) | Apparent power `|S|` (MVA) |
| Linearity | Exact linear map `Delta_f = H * Delta_p` | Linearized via Jacobian |
| Variables | `theta` only | `theta` and `V` (PQ buses) |
| Sensitivity | Direct PTDF matrix `H` | Adjoint solve `h = J^{-T} b` |
| Limits | `|f| <= c` (MW) | `|S| <= c` (MVA) at both ends |
| Balanced subspace dim. | `n - 1` | `(n - 1) + n_pq` |
| Per-line constraints | 1 (signed flow) | 2 (from-end and to-end) |

### 13.2 L2 vs. Sigma vs. Metric Norms

| Aspect | L2 Radius | Sigma-Radius | Metric Radius |
|--------|-----------|--------------|---------------|
| Norm | `||Delta_p||_2` | `||Sigma^{-1/2} Delta_p||_2` | `||Delta_p||_M` |
| Interpretation | Worst-case over unit ball | In units of std devs | Under arbitrary SPD metric |
| Denominator | `||g||_2` | `||Sigma^{1/2} g||_2` | `sqrt(g^T M^{-1} g)` |
| Extra output | -- | Overload probability | -- |
| Special case of Metric | `M = I` | `M = Sigma^{-1}` | General `M` |
| Balance projection | Unweighted mean | `sigma^2`-weighted mean | Unweighted mean |

### 13.3 Relationships Between Radii

All radius formulations share the common structure:

```
r_l = margin_l / denominator_l
```

where:

```
margin_l = c_l - |flow_0_l|          (MW or MVA)
denominator_l = dual norm of g_l     (depends on chosen metric)
```

The sigma-radius and L2 radius are connected by:

```
r_sigma = r_L2 * (||g||_2 / sigma_flow)
```

In general, for metric `M`:

```
r_M = margin / sqrt(g^T M^{-1} g)
```

with special cases:
- `M = I` yields `r_L2`
- `M = diag(1/sigma^2)` yields `r_sigma`

---

## 14. Summary Comparison Table

| Formulation | Model | Norm | Sensitivity | Margin | Balance | Probabilistic | Implementation |
|-------------|-------|------|-------------|--------|---------|---------------|----------------|
| DC L2 | DC | L2 | `g_l = H[l,:]` | `c - \|f_0\|` (MW) | `Proj(g)` | No | `core_l2.py` |
| DC Sigma | DC | Sigma | `g_l = H[l,:]` | `c - \|f_0\|` (MW) | N/A | Yes (Gaussian) | `probabilistic.py` |
| DC Metric | DC | M-norm | `g_l = H[l,:]` | `c - \|f_0\|` (MW) | N/A | No | `metric.py` |
| DC N-1 | DC | L2 | `g^{(k)} = g + LODF*g_k` | Post-cont. | `Proj(g^{(k)})` | No | `nminus1.py` |
| AC L2 | AC | L2 | `h_l = J^{-T} b_l` | `c - \|S_0\|` (MVA) | 2-block `Proj` | No | `ac_l2.py` |
| AC Sigma | AC | Sigma | `h_l = J^{-T} b_l` | `c - \|S_0\|` (MVA) | `sigma^2`-weighted | Yes (Gaussian) | `ac_sigma_radius.py` |
| AC Metric | AC | M-norm | `h_l = J^{-T} b_l` | `c - \|S_0\|` (MVA) | 2-block `Proj` | No | `ac_metric_radius.py` |
| Baselines | AC | -- | -- | `c - \|S_0\|` (MVA) | -- | Cantelli bound | `ac_baselines.py` |

---

## 15. Implementation File Map

| File | Description |
|------|-------------|
| `src/stability_radius/dc/dc_model.py` | DC model: `B` matrix, PTDF matrix `H`, `DCOperator` with LU-cached solves |
| `src/stability_radius/ac/ac_model.py` | AC model: `Ybus`, Jacobian `J`, `ACOperator` with adjoint solves |
| `src/stability_radius/radii/core_l2.py` | Pure L2 certificate: margins, norms, radii, balanced projection helpers |
| `src/stability_radius/radii/l2.py` | High-level DC L2 radius wrapper |
| `src/stability_radius/radii/probabilistic.py` | DC sigma-radius and Gaussian overload probability |
| `src/stability_radius/radii/metric.py` | DC metric radius with Cholesky-based computation |
| `src/stability_radius/radii/nminus1.py` | N-1 contingency radius with LODF computation |
| `src/stability_radius/radii/ac_l2.py` | AC L2 radius with adjoint sensitivities and from/to aggregation |
| `src/stability_radius/radii/ac_sigma_radius.py` | AC sigma-radius with sigma-squared-weighted balance |
| `src/stability_radius/radii/ac_metric_radius.py` | AC metric radius with diagonal and dense `M` support |
| `src/stability_radius/metrics/ac_baselines.py` | Baseline metrics: loading ratio, headroom, Cantelli bound |
| `src/stability_radius/radii/common.py` | Shared utilities: limit estimation, line key formatting |
