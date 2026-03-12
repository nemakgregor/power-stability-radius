# Algorithms and Models

This document provides comprehensive algorithmic documentation for every solver,
computation pipeline, and verification workflow in the power-stability-radius
project. Each section covers purpose, inputs/outputs, step-by-step logic,
key parameters, implementation references, computational complexity, and
design rationale.

For mathematical foundations (norms, projections, Cauchy--Schwarz certificates),
see [mathematical_foundations.md](mathematical_foundations.md).

---

## Table of Contents

1. [DC Operator Construction](#1-dc-operator-construction)
2. [AC Operator Construction](#2-ac-operator-construction)
3. [DC L2 Radius Computation](#3-dc-l2-radius-computation)
4. [AC L2 Radius Computation](#4-ac-l2-radius-computation)
5. [AC Sigma-Radius Computation](#5-ac-sigma-radius-computation)
6. [AC Metric-Radius Computation](#6-ac-metric-radius-computation)
7. [Base Point Computation Pipeline](#7-base-point-computation-pipeline)
8. [Monte Carlo Verification](#8-monte-carlo-verification)
9. [Certificate Verification](#9-certificate-verification)
10. [Metrics Analysis Pipeline](#10-metrics-analysis-pipeline)
11. [N-1 Contingency Radius](#11-n-1-contingency-radius)

---

## 1. DC Operator Construction

**Name:** `build_dc_operator`

**Purpose:** Build a sparse DC linear operator from a pandapower network that
supports PTDF-like sensitivity computations without materializing the full dense
sensitivity matrix H. The operator encapsulates the reduced nodal susceptance
matrix, its LU factorization, and line-flow projection matrices, enabling
efficient incremental flow computations.

**Implementation file:**
`src/stability_radius/dc/dc_model.py`, function `build_dc_operator` (line 626)

### Inputs

| Parameter   | Type                | Description                              |
|-------------|---------------------|------------------------------------------|
| `net`       | pandapower Network  | Pandapower network with bus, line, trafo, impedance tables |
| `slack_bus` | `int` (default `0`) | Slack bus ID or position in sorted bus order |

### Output

| Type         | Description |
|--------------|-------------|
| `DCOperator` | Frozen dataclass containing bus/line IDs, LU factorization, projection matrices, and phase-shift injection terms |

### DCOperator Fields

- `bus_ids`: Sorted tuple of bus IDs
- `line_ids`: Sorted tuple of monitored line IDs
- `slack_pos`: Integer position of the slack bus
- `from_bus_pos`, `to_bus_pos`: Arrays mapping each line to bus positions
- `b`: Per-line branch susceptance coefficients (MW/rad)
- `mask_non_slack`: Boolean mask excluding the slack bus
- `red_pos_of_bus_pos`: Maps full bus position to reduced (slack-eliminated) position
- `Bred_lu`: SciPy SuperLU factorization of the reduced susceptance matrix
- `W`: Sparse matrix `A_lines_red * diag(b_lines)` for flow computation
- `shift_inj_red`: Constant RHS vector from phase-shifting transformers

### Algorithm (Step-by-Step)

```
PROCEDURE build_dc_operator(net, slack_bus):
    1. PARSE NETWORK TOPOLOGY
       bus_ids <- sorted(net.bus.index)
       n_bus <- |bus_ids|
       slack_pos <- resolve_slack_pos(bus_ids, slack_bus)
       line_ids <- sorted(net.line.index)
       m_line <- |line_ids|
       bus_pos <- {bid: pos for pos, bid in enumerate(bus_ids)}

    2. BUILD MONITORED LINE INCIDENCE AND COEFFICIENTS
       FOR each line lid in line_ids:
           row <- net.line.loc[lid]
           fpos <- bus_pos[from_bus]
           tpos <- bus_pos[to_bus]
           IF in_service:
               x_total <- x_ohm_per_km * length_km / parallel
               vn_kv <- bus_vn_kv(net, from_bus)
               b_line <- vn_kv^2 / x_total       # MW/rad
               A_lines[lid_pos, fpos] <- +1
               A_lines[lid_pos, tpos] <- -1

    3. BUILD FULL BRANCH SET FOR B MATRIX
       (includes lines, transformers, impedances)

       a. Lines: same b as above, shift_rad = 0

       b. Transformers:
           FOR each trafo tid in sorted(net.trafo.index):
               x_ohm <- trafo_x_total_ohm(net, row)
                 z_pu <- vk_percent / 100
                 r_pu <- vkr_percent / 100
                 x_pu <- sqrt(max(z_pu^2 - r_pu^2, 0))
                 Z_base <- vn_hv_kv^2 / sn_mva
                 x_ohm <- x_pu * Z_base
               b_raw <- vn_hv_kv^2 / x_ohm
               tap <- trafo_tap_ratio(row)
                 tap = 1 + (tap_pos - tap_neutral) * tap_step_percent / 100
                 IF tap_side == "lv": tap <- 1/tap
               b_eff <- b_raw / tap
               shift_rad <- shift_degree * pi / 180
               ADD branch(hv_pos, lv_pos, b_eff, shift_rad)

       c. Impedances:
           FOR each impedance iid:
               x_pu <- xft_pu
               Z_base <- vn_kv^2 / sn_mva
               x_ohm <- x_pu * Z_base
               b_imp <- vn_kv^2 / x_ohm
               ADD branch(from_pos, to_pos, b_imp, 0)

    4. CONNECTIVITY CHECK (union-find)
       Verify all buses connected to slack via undirected edges.
       RAISE ValueError if disconnected buses exist.

    5. ASSEMBLE REDUCED SUSCEPTANCE MATRIX
       A_all <- sparse incidence (m_all x n_bus)
       mask_non_slack <- boolean mask excluding slack_pos
       A_all_red <- A_all[:, mask_non_slack]      # (m_all x n-1)
       W_all <- A_all_red * diag(b_all)           # (m_all x n-1)
       B_red <- A_all_red^T @ W_all               # (n-1 x n-1), CSC

    6. BUILD LINE FLOW PROJECTION MATRIX
       A_lines_red <- A_lines[:, mask_non_slack]   # (m_line x n-1)
       W <- A_lines_red * diag(b_lines)            # (m_line x n-1)

    7. COMPUTE PHASE-SHIFTER INJECTION TERM
       b_shift <- b_all * shift_all_rad            # (m_all,)
       shift_inj_red <- A_all_red^T @ b_shift      # (n-1,)

    8. LU FACTORIZE B_red
       Bred_lu <- scipy.sparse.linalg.splu(B_red)

    9. RETURN DCOperator(...)
```

### Key Mathematical Relationships

The DC power flow model satisfies:

- `B * theta = p + A^T * diag(b) * shift`
- `f_lines = W * theta_red` (monitored line flows)
- For perturbations: `delta_f = H * delta_p`, where `H = W * B_red^{-1}` (restricted to non-slack buses)

The sensitivity of line `l` to injection perturbations is computed via the
adjoint solve `g_l = B_red^{-1} * w_l` where `w_l` is the l-th row of W
transposed.

### Computational Complexity

| Step | Complexity |
|------|-----------|
| Network parsing | O(m + n) |
| Incidence matrix assembly | O(m) |
| B_red assembly (sparse) | O(m) |
| LU factorization | O(n^{1.5}) typical for sparse grids |
| Per-chunk sensitivity solve | O(n * chunk_size) via LU back-substitution |
| Full H materialization | O(m * n) |

### Design Rationale

- **Lazy sensitivity computation:** The operator avoids materializing the full
  H matrix (m x n), instead computing sensitivity rows on-demand via LU
  back-substitution. This is critical for large networks where H may not fit
  in memory.
- **Union-find connectivity:** A lightweight O(m * alpha(n)) check ensures the
  reduced system is non-singular before attempting LU factorization, producing
  clear diagnostic messages instead of opaque LAPACK errors.
- **Phase-shifter support:** The constant injection term `shift_inj_red` is
  separated from the perturbation-dependent computation, since phase shifts
  are fixed network parameters that cancel in sensitivity (delta) computations.

---

## 2. AC Operator Construction

**Name:** `build_ac_operator`

**Purpose:** Build a sparse AC power-flow Jacobian-based linear operator around
a given AC operating point (base voltages), supporting adjoint sensitivity
computations for apparent power flow constraints. The operator handles PV/PQ
bus classification and provides efficient transposed Jacobian solves.

**Implementation file:**
`src/stability_radius/ac/ac_model.py`, function `build_ac_operator` (line 672)

### Inputs

| Parameter       | Type              | Description |
|-----------------|-------------------|-------------|
| `net`           | pandapower Network | Network topology and parameters |
| `slack_bus`     | `int`             | Slack bus ID or position |
| `vm_pu`         | `ndarray (n,)`    | Base voltage magnitudes (per-unit) |
| `va_rad`        | `ndarray (n,)`    | Base voltage angles (radians) |
| `line_indices`  | `list[int]` or `None` | Explicit monitored line ordering |
| `lossless`      | `bool` (default `True`) | If True, enforce r=0 in Ybus |

### Output

| Type         | Description |
|--------------|-------------|
| `ACOperator` | Frozen dataclass with Ybus, Jacobian, LU factorization, PV/PQ masks, and per-line admittances |

### Algorithm (Step-by-Step)

```
PROCEDURE build_ac_operator(net, slack_bus, vm_pu, va_rad, ...):
    1. PARSE TOPOLOGY
       bus_ids <- sorted(net.bus.index)
       n_bus <- |bus_ids|
       slack_pos <- resolve_slack_pos(bus_ids, slack_bus)

    2. DETECT PV BUSES
       pv_mask <- boolean array (n_bus,)
       FOR each in-service gen/ext_grid (excluding slack):
           pv_mask[bus_pos[gen_bus]] <- True
       pv_mask[slack_pos] <- False   # slack excluded from reduced system

    3. BUILD SPARSE Ybus (per-unit, series-only)
       FOR each in-service line:
           z_ohm <- r + jx  (r=0 if lossless)
           z_pu <- z_ohm / Z_base
           y <- 1/z_pu
           Ybus[i,i] += y;  Ybus[k,k] += y
           Ybus[i,k] -= y;  Ybus[k,i] -= y

       FOR each in-service transformer:
           z_pu <- j * x_ohm / Z_base
           y <- 1/z_pu
           a <- tap * exp(j * phi)     # complex tap ratio
           Ybus stamping with tap:
             Y_ii += y/|a|^2;  Y_kk += y
             Y_ik -= y/conj(a); Y_ki -= y/a

       FOR each in-service impedance:
           z_pu <- rft_pu + j*xft_pu
           y <- 1/z_pu
           Standard Ybus stamping (no tap)

    4. COMPUTE BASE POWER INJECTIONS
       V <- vm * exp(j * va)
       I <- Ybus @ V
       S <- V * conj(I)
       P <- Re(S),  Q <- Im(S)     # per-unit

    5. BUILD REDUCED PF JACOBIAN
       Structure (with n_theta non-slack, n_pq PQ-only):
         [dP/dtheta  dP/dV_pq]   rows: P for all non-slack
         [dQ/dtheta  dQ/dV_pq]   rows: Q for PQ buses only

       Off-diagonal elements (for Ybus entry Y_ik, i != k):
         dP_i/dtheta_k = Vi*Vk*(G*sin(theta_ik) - B*cos(theta_ik)) * sn_mva
         dP_i/dV_k     = Vi*(G*cos(theta_ik) + B*sin(theta_ik)) * sn_mva
         dQ_i/dtheta_k = -Vi*Vk*(G*cos(theta_ik) + B*sin(theta_ik)) * sn_mva
         dQ_i/dV_k     = Vi*(G*sin(theta_ik) - B*cos(theta_ik)) * sn_mva

       Diagonal elements:
         dP_i/dtheta_i = (-Q_i - B_ii*Vi^2) * sn_mva
         dP_i/dV_i     = (P_i/Vi + G_ii*Vi) * sn_mva
         dQ_i/dtheta_i = (P_i - G_ii*Vi^2) * sn_mva
         dQ_i/dV_i     = (Q_i/Vi - B_ii*Vi) * sn_mva

       PV bus handling:
         - V_magnitude is NOT a free variable for PV buses
         - Q equation is excluded for PV buses
         - Variables: x = [theta_non_slack; V_pq]
         - Equations: [P_non_slack; Q_pq]

    6. LU FACTORIZE JACOBIAN
       J_lu <- scipy.sparse.linalg.splu(J)

    7. STORE PER-LINE ADMITTANCES
       FOR each monitored line:
           y_series_pu[pos] <- 1 / z_pu  (complex)

    8. RETURN ACOperator(...)
```

### Key Operations

- **Forward solve:** `J * dx = du` via `J_lu.solve(rhs, trans="N")`
- **Adjoint solve:** `J^T * a = b` via `J_lu.solve(rhs, trans="T")`

The adjoint solve is the core operation for computing per-line sensitivity
vectors (h-vectors) used in all AC radius calculations.

### Computational Complexity

| Step | Complexity |
|------|-----------|
| Ybus assembly | O(m) where m = branches |
| Jacobian assembly | O(nnz(Ybus)) |
| LU factorization | O(n_vars^{1.5}) for sparse grids |
| Per adjoint solve | O(n_vars) via LU back-substitution |

### Design Rationale

- **Lossless default:** Setting r=0 keeps the AC linearization aligned with
  the project's DC convention, ensuring consistency between DC and AC
  certificates. The Jacobian is still computed at the full AC operating point.
- **PV/PQ bus handling:** Generators with voltage control (PV buses) have fixed
  voltage magnitude, reducing the Jacobian dimension and ensuring the
  certificate correctly reflects which variables respond to injection
  perturbations.
- **Series-only Ybus:** Shunt elements are excluded because the project's
  certificate concerns injection perturbations, and shunt admittances are
  constant network parameters that do not change with injections.

---

## 3. DC L2 Radius Computation

**Name:** `compute_l2_radius`

**Purpose:** Compute a per-line L2 robustness radius using the DC power transfer
distribution factor (PTDF) sensitivity matrix. The radius measures the maximum
L2-norm injection perturbation that guarantees no line flow exceeds its thermal
limit under the lossless DC power flow model.

**Implementation files:**
- `src/stability_radius/radii/l2.py`, function `compute_l2_radius` (line 19)
- `src/stability_radius/radii/core_l2.py`, helper functions (lines 56, 84)

### Inputs

| Parameter      | Type                    | Description |
|----------------|-------------------------|-------------|
| `net`          | pandapower Network      | Network for limit extraction |
| `H_full`       | `ndarray (m, n)`        | PTDF sensitivity matrix |
| `limit_factor` | `float` (default `1.0`) | Multiplier on thermal limits |
| `base`         | `LineBaseQuantities` or `None` | Precomputed base quantities |

### Output

| Type                              | Description |
|-----------------------------------|-------------|
| `Dict[str, Dict[str, Any]]`      | `"line_{idx}"` -> metrics dict with keys: `flow0_mw`, `p0_mw`, `p_limit_mw_est`, `is_unconstrained`, `margin_mw`, `norm_g`, `radius_l2` |

### Algorithm

```
PROCEDURE compute_l2_radius(net, H_full, limit_factor, base):
    1. EXTRACT BASE QUANTITIES
       IF base is None:
           base_q <- get_line_base_quantities(net, limit_factor)
       (This runs DC OPF to obtain base flows and limits)

    2. COMPUTE PROJECTED ROW NORMS (balanced disturbances)
       FOR each line l (row of H):
           g_l <- H[l, :]
           # Project onto balanced subspace {delta_p : sum(delta_p) = 0}
           norm_g_l <- ||g_l - mean(g_l) * 1||_2
                     = sqrt(||g_l||^2 - (sum(g_l))^2 / n)

    3. COMPUTE PER-LINE RADIUS
       FOR each line l:
           margin_l <- limit_l - |flow0_l|
           IF norm_g_l > 1e-12:
               r_l <- margin_l / norm_g_l
           ELSE:
               r_l <- +inf    # line insensitive on balanced subspace

    4. RETURN results dict
```

### Key Parameters

- `limit_factor`: Scaling applied to thermal limits (default 1.0 = no scaling)
- Balanced projection: The norm `||Proj(g)||_2` is used rather than `||g||_2`
  because injections satisfy `sum(delta_p) = 0` (power balance constraint)

### Certificate Guarantee

For any balanced perturbation `delta_p` with `||delta_p||_2 <= r*`:

```
|f_l + g_l^T * delta_p| <= |f0_l| + ||Proj(g_l)||_2 * ||delta_p||_2
                         <= |f0_l| + margin_l
                         = c_l
```

by the Cauchy-Schwarz inequality restricted to the balanced subspace.

### Computational Complexity

O(m * n) for the projected norm computation over all m lines with n buses.

### Design Rationale

- **Balanced projection:** Using the ones-complement projection
  `Proj(g) = g - mean(g) * 1` is essential because DC sensitivities are only
  defined up to an additive constant (the slack bus reference). Without
  projection, different slack bus choices would yield different radii.

---

## 4. AC L2 Radius Computation

**Name:** `compute_ac_l2_radius`

**Purpose:** Compute a fast AC L2 stability radius certificate around an AC
power flow base point. Unlike the DC version, this operates on apparent power
`|S| = sqrt(P^2 + Q^2)` constraints with two endpoints per line (from-end
and to-end), selecting the binding (smaller radius) end.

**Implementation file:**
`src/stability_radius/radii/ac_l2.py`, function `compute_ac_l2_radius` (line 59)

### Inputs

| Parameter         | Type                | Description |
|-------------------|---------------------|-------------|
| `net`             | pandapower Network  | Network topology |
| `base_pf`         | `PyPSAAPFResult`    | AC PF base point (V, theta, P, Q per line) |
| `slack_bus`        | `int`               | Slack bus ID |
| `chunk_size`       | `int` (default 256) | Lines per LU-solve batch |
| `balance`          | `bool` (default True) | Use balanced projection |
| `lossless`         | `bool` (default True) | Lossless Ybus |
| `return_h_vectors` | `bool` (default False) | Store sensitivity vectors |

### Output

| Type                              | Description |
|-----------------------------------|-------------|
| `Dict[str, Dict[str, Any]]`      | Per-line results with from/to end metrics, aggregate radius, binding end, margin, sensitivity norm |

### Algorithm

```
PROCEDURE compute_ac_l2_radius(net, base_pf, slack_bus, ...):
    1. BUILD AC OPERATOR
       op <- build_ac_operator(net, slack_bus, vm_pu, va_rad, lossless)

    2. EXTRACT LIMITS AND BASE FLOWS
       FOR each line l:
           limits_mva[l] <- estimate_line_limit_mva_with_flag(net, line_row)
           s0_from[l] <- sqrt(p0[l]^2 + q0[l]^2)
           s0_to[l]   <- sqrt(p1[l]^2 + q1[l]^2)
           margin_from[l] <- limit[l] - s0_from[l]
           margin_to[l]   <- limit[l] - s0_to[l]

    3. CONSTRUCT ADJOINT RHS VECTORS (chunked)
       n_constraints <- 2 * m_lines  (from-end and to-end)
       FOR each constraint con_idx in chunks of chunk_size:
           line_pos <- con_idx // 2
           is_from <- (con_idx % 2) == 0
           i_pos <- from_bus if is_from else to_bus
           k_pos <- to_bus if is_from else from_bus

           # Compute branch flow derivatives (per-unit, scaled by sn_mva)
           y <- y_series_pu[line_pos]
           theta <- va[i] - va[k]
           A <- g*cos(theta) + b*sin(theta)
           B_tmp <- g*sin(theta) - b*cos(theta)

           dP/dtheta_i = Vi*Vk*B_tmp * sn_mva
           dP/dtheta_k = -dP/dtheta_i
           dQ/dtheta_i = -Vi*Vk*A * sn_mva
           dQ/dtheta_k = -dQ/dtheta_i
           dP/dVi = (2g*Vi - Vk*A) * sn_mva
           dP/dVk = -Vi*A * sn_mva
           dQ/dVi = (-2b*Vi - Vk*B_tmp) * sn_mva
           dQ/dVk = -Vi*B_tmp * sn_mva

           # Chain rule for |S| gradient
           s0 <- apparent power at this end
           IF s0 > eps:
               wP <- P / s0     # d|S|/dP = P/|S|
               wQ <- Q / s0     # d|S|/dQ = Q/|S|
           ELSE:
               wP <- wQ <- 1/sqrt(2)   # conservative fallback

           # Assemble adjoint RHS b = d|S|/dx
           b_theta_i <- wP * dP/dtheta_i + wQ * dQ/dtheta_i
           b_theta_k <- wP * dP/dtheta_k + wQ * dQ/dtheta_k
           b_Vi <- wP * dP/dVi + wQ * dQ/dVi
           b_Vk <- wP * dP/dVk + wQ * dQ/dVk

           # Place into RHS matrix B at correct positions
           B[theta_red_pos[i], j] += b_theta_i
           B[theta_red_pos[k], j] += b_theta_k
           B[n_theta + v_red_pos[i], j] += b_Vi    # only if PQ bus
           B[n_theta + v_red_pos[k], j] += b_Vk    # only if PQ bus

    4. SOLVE ADJOINT SYSTEM (chunked)
       Y <- J_lu.solve(B, trans="T")   # J^T * Y = B

    5. COMPUTE NORMS AND RADII
       FOR each constraint:
           a_p <- Y[0:n_theta, j]    # P-block of adjoint
           a_q <- Y[n_theta:, j]     # Q-block of adjoint

           IF balance:
               # Balanced two-block norm:
               # sqrt(||Proj_P(a_p)||^2 + ||Proj_Q(a_q)||^2)
               # where Proj_P subtracts mean over n_bus entries
               # and Proj_Q subtracts mean over n_pq entries
               denom <- balanced_two_block_norm(a_p, a_q, n_bus, n_pq)
           ELSE:
               denom <- ||Y[:, j]||_2

           radius <- margin / denom   (or inf if denom < eps)

    6. AGGREGATE PER LINE
       FOR each line l:
           r_line <- min(r_from, r_to)
           binding_end <- "from" if r_from <= r_to else "to"

    7. RETURN results dict (optionally including h_vectors)
```

### Key Parameters

| Parameter    | Effect |
|-------------|--------|
| `chunk_size` | Number of constraints per LU solve batch (memory vs. speed) |
| `balance`    | Enables balanced projection (sum-zero P and Q) |
| `lossless`   | Enforces r=0 in Ybus for DC alignment |
| `return_h_vectors` | Stores full adjoint vectors for sigma/metric radius |

### Balanced Two-Block Norm

For AC injections with constraints `1^T * delta_P = 0` and `1^T * delta_Q = 0`:

```
||a||_bal = sqrt( ||Proj_P(a_P)||^2 + ||Proj_Q(a_Q)||^2 )

where:
  Proj_P(v) = v - (sum(v)/n_bus) * 1
  Proj_Q(v) = v - (sum(v)/n_pq) * 1
```

The P-block is projected over all buses (n_bus), while the Q-block is projected
over PQ buses only (n_pq), because PV buses do not have Q as a free variable.

### Computational Complexity

| Step | Complexity |
|------|-----------|
| ACOperator build | O(nnz(Ybus) + n_vars^{1.5}) |
| Adjoint solves | O(2m * n_vars) via chunked LU back-substitution |
| Norm computation | O(2m * n_vars) |
| Total | O(m * n) dominated by adjoint solves |

### Design Rationale

- **Two-end constraints:** Unlike DC where flow direction is unambiguous,
  AC apparent power differs at the from-end and to-end of a line due to
  losses and reactive power. Both must be checked.
- **Chain rule for |S|:** The gradient of `|S| = sqrt(P^2 + Q^2)` is computed
  analytically via `d|S|/dP = P/|S|`, `d|S|/dQ = Q/|S|`, avoiding numerical
  differentiation.
- **Fallback at |S|=0:** When base apparent power is zero, the gradient is
  undefined. The equal-weight fallback `(1/sqrt(2), 1/sqrt(2))` provides a
  conservative, unbiased certificate direction.

---

## 5. AC Sigma-Radius Computation

**Name:** `compute_ac_sigma_radius`

**Purpose:** Compute a probabilistic stability radius measured in
"number of standard deviations" units, using precomputed adjoint h-vectors
and per-bus injection standard deviations. This enables heterogeneous
uncertainty modeling where different buses have different injection variability.

**Implementation file:**
`src/stability_radius/radii/ac_sigma_radius.py`, function `compute_ac_sigma_radius` (line 159)

### Inputs

| Parameter        | Type              | Description |
|------------------|-------------------|-------------|
| `h_vectors`      | `ndarray (m, 2n)` | Adjoint sensitivity vectors from AC L2 |
| `s_limit_mva`    | `ndarray (m,)`     | Thermal limits per line (MVA) |
| `s0_mva`         | `ndarray (m,)`     | Base apparent power per line (MVA) |
| `sigma_p_mw`     | `ndarray (n,)`     | Per-bus active power std dev (MW) |
| `sigma_q_mvar`   | `ndarray (n,)`     | Per-bus reactive power std dev (MVAr) |
| `line_ids`       | `Sequence[int]` or `None` | Line indices for keys |
| `balance`        | `bool` (default True) | Apply balanced projection |
| `eps_sigma_flow`  | `float` (default 1e-15) | Zero-sensitivity threshold |

### Output

| Type                              | Description |
|-----------------------------------|-------------|
| `dict[str, dict[str, Any]]`      | Per-line: `sigma_flow_mva`, `radius_ac_sigma`, `overload_probability_ac`, `worst_case_dp_mw`, `worst_case_dq_mvar`, `worst_case_s_predicted_mva` |

### Algorithm

```
PROCEDURE compute_ac_sigma_radius(h_vectors, sigma_p, sigma_q, ...):
    1. VALIDATE AND RESHAPE INPUTS
       H <- (n_lines x 2*n_bus) matrix
       hP <- H[:, 0:n_bus]     # P-block
       hQ <- H[:, n_bus:2n]    # Q-block

    2. OPTIONAL BALANCED PROJECTION (sigma^2-weighted)
       IF balance:
           # Enforce sum(dp) = 0 for worst-case perturbation
           # Requires sum(sigma_p^2 * hP) = 0
           mu_p <- sum(sigma_p^2 * hP, axis=1) / sum(sigma_p^2)
           hP <- hP - mu_p
           mu_q <- sum(sigma_q^2 * hQ, axis=1) / sum(sigma_q^2)
           hQ <- hQ - mu_q

    3. COMPUTE SIGMA-WEIGHTED FLOW SENSITIVITY
       scaledP <- hP * sigma_p          # element-wise (broadcasting)
       scaledQ <- hQ * sigma_q
       sigma_flow[l] <- sqrt(sum(scaledP[l]^2) + sum(scaledQ[l]^2))
                       = ||[sigma_p * hP_l, sigma_q * hQ_l]||_2

    4. COMPUTE RADII
       margin <- c - s0
       radius[valid] <- margin / sigma_flow   (where sigma_flow > eps)
       radius[degenerate] <- +inf if margin >= 0, else -inf

    5. COMPUTE WORST-CASE PERTURBATIONS
       dp[l] <- radius[l] * sigma_p^2 * hP[l] / sigma_flow[l]
       dq[l] <- radius[l] * sigma_q^2 * hQ[l] / sigma_flow[l]

    6. COMPUTE PREDICTED WORST-CASE |S|
       s_pred[l] <- s0[l] + hP[l]^T * dp[l] + hQ[l]^T * dq[l]
       (For valid lines, this equals c[l] up to numerical noise)

    7. COMPUTE OVERLOAD PROBABILITIES (Gaussian Q-function)
       FOR each line l:
           P(|S| > c) = Q((c - s0) / sigma_flow) + Q((c + s0) / sigma_flow)
           where Q(x) = 0.5 * erfc(x / sqrt(2))

    8. RETURN results dict
```

### Key Mathematical Relationships

Given diagonal covariance `Sigma = diag(sigma_p^2, sigma_q^2)`:

- **Flow standard deviation:** `sigma_flow = ||Sigma^{1/2} * h||_2`
- **Sigma-radius:** `r_sigma = (c - |S0|) / sigma_flow` (dimensionless, in sigma units)
- **Worst-case perturbation:** `delta_u* = r_sigma * Sigma * h / sigma_flow`
- **Overload probability:** Based on linearized Gaussian model for |S|

### Balanced Projection (sigma^2-weighted)

The balanced projection differs from the L2 case because the perturbation
ellipsoid is anisotropic. The constraint `sum(dp) = 0` with
`dp_i = r * sigma_i^2 * hP_i / sigma_flow` requires
`sum(sigma_i^2 * hP_i) = 0`. This is achieved by:

```
hP_adj = hP - sum(sigma_P^2 * hP) / sum(sigma_P^2)
```

### Computational Complexity

O(m * n) -- linear in lines and buses. No matrix solves needed (h-vectors
are precomputed).

### Design Rationale

- **Stateless module:** Intentionally decoupled from ACOperator construction
  to allow flexible uncertainty modeling without rebuilding the Jacobian.
- **Per-bus sigma:** Enables heterogeneous uncertainty (e.g., high-variability
  renewable buses vs. stable conventional buses).
- **Gaussian Q-function:** Provides an analytic overload probability estimate
  that can be compared against Monte Carlo empirical rates.

---

## 6. AC Metric-Radius Computation

**Name:** `compute_ac_metric_radius`

**Purpose:** Compute a generalized stability radius under an arbitrary symmetric
positive definite (SPD) weight matrix M. This subsumes both the L2 radius
(M = I) and the sigma-radius (M = Sigma^{-1}) as special cases, providing a
unified framework for custom disturbance metrics.

**Implementation file:**
`src/stability_radius/radii/ac_metric_radius.py`, function `compute_ac_metric_radius` (line 96)

### Inputs

| Parameter     | Type              | Description |
|---------------|-------------------|-------------|
| `h_vectors`   | `ndarray (m, 2n)` | Adjoint sensitivity vectors |
| `s_limit_mva` | `ndarray (m,)`    | Thermal limits (MVA) |
| `s0_mva`      | `ndarray (m,)`    | Base apparent power (MVA) |
| `M`           | `ndarray (2n,)` or `(2n, 2n)` | SPD weight matrix (diagonal or dense) |
| `line_ids`    | `Sequence[int]` or `None` | Line indices |
| `balance`     | `bool` (default True) | Balanced projection |
| `eps_denom`   | `float` (default 1e-12) | Zero threshold |

### Output

| Type                              | Description |
|-----------------------------------|-------------|
| `dict[str, dict[str, Any]]`      | Per-line: `metric_denom`, `margin_mva`, `radius_ac_metric` |

### Algorithm

```
PROCEDURE compute_ac_metric_radius(h_vectors, s_limit, s0, M, ...):
    1. VALIDATE INPUTS
       H <- (m x 2n), M <- (2n,) or (2n x 2n)

    2. OPTIONAL BALANCED PROJECTION
       IF balance:
           hP <- H[:, :n] - mean(H[:, :n], axis=1)
           hQ <- H[:, n:] - mean(H[:, n:], axis=1)
           H_proj <- [hP | hQ]

    3. COMPUTE METRIC DENOMINATOR
       IF M is diagonal (1-D):
           # O(n) per line
           denom[l] <- sqrt(sum(H_proj[l]^2 / M))
       ELSE (dense):
           # Cholesky: L L^T = M
           L <- cholesky(M)
           Z <- L^{-1} H_proj^T      # solve L*Z = H^T
           denom[l] <- ||Z[:, l]||_2

    4. COMPUTE RADII
       margin <- c - s0
       radius[valid] <- margin / denom
       radius[degenerate] <- +inf / -inf

    5. RETURN results dict
```

### Mathematical Definition

```
r_l^M = (c_l - |S_l^0|) / sqrt(h_l^T * M^{-1} * h_l)
```

### Computational Complexity

| M type   | Complexity |
|----------|-----------|
| Diagonal | O(m * n) |
| Dense    | O(n^3) for Cholesky + O(m * n^2) for solves |

### Design Rationale

- **Diagonal optimization:** When M is diagonal (common case: diagonal
  covariance), the Cholesky factorization is avoided entirely, reducing
  complexity from O(n^3) to O(n).
- **Unified framework:** A single implementation covers L2, sigma, and
  arbitrary metric radii, reducing code duplication and testing burden.

---

## 7. Base Point Computation Pipeline

The project supports multiple strategies for computing the operating point
(base point) around which stability radii are linearized.

### 7.1 Case Dispatch

**Purpose:** Use the dispatch directly from the MATPOWER/PGLib case file
without solving any optimization problem.

**Implementation:** Handled by reading `net.bus` injections and line flows
from the parsed pandapower network.

### 7.2 DC OPF via PyPSA + HiGHS

**Name:** `solve_dc_opf_base_flows_from_pandapower`

**Purpose:** Solve a single-snapshot DC optimal power flow to obtain base
line flows and generator dispatch.

**Implementation file:**
`src/stability_radius/base_point/pypsa_opf.py`, function
`solve_dc_opf_base_flows_from_pandapower` (line 156)

#### Algorithm

```
PROCEDURE solve_dc_opf(net, line_indices, line_limits_mw, opf_cfg):
    1. CREATE PyPSA NETWORK
       n <- pypsa.Network()
       Set snapshots, sn_mva, carrier

    2. ADD BUSES
       FOR each bus in sorted(net.bus.index):
           n.add("Bus", str(bus_id), v_nom=vn_kv)

    3. ADD LOADS
       Aggregate P per bus from net.load + net.shunt

    4. ADD GENERATORS (merit order dispatch)
       FOR each in-service gen in net.gen:
           p_nom <- max_p_mw
           p_min_pu <- min_p_mw / max_p_mw
           marginal_cost <- rank_order  (ascending)
       FOR each in-service sgen in net.sgen:
           (same treatment as gen)
       FOR each in-service ext_grid:
           Two components per ext_grid:
             ext_{eid}        — generation (sign=+1, high cost)
             ext_{eid}_absorb — absorption (sign=-1, higher cost)

    5. ADD LINES AND TRANSFORMERS
       FOR each monitored line:
           n.add("Line", r=0, x=x_ohm, s_nom=limit_mw)
       FOR each transformer:
           Convert to PyPSA Transformer with tap + phase shift
           x_pu_dc = x_pu * scale * tap  (MATPOWER DC convention)

    6. SOLVE DC OPF
       n.optimize(solver_name="highs", solver_options=...)

    7. EXTRACT RESULTS
       line_flows <- n.lines_t.p0[snapshot]
       bus_injections <- gen_by_bus - load_by_bus

    8. RETURN PyPSAOPFResult(line_flows, bus_injections, status, objective)
```

#### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `solver_name` | `"highs"` | LP solver (project policy: HiGHS only) |
| `headroom_factor` | from config | Scales line limits in OPF to create margin |
| `unconstrained_line_nom_mw` | from config | Surrogate limit for unconstrained lines |

### 7.3 AC Power Flow via pandapower.runpp()

**Name:** `solve_ac_pf_base_point_from_pandapower`

**Purpose:** Solve an AC power flow to obtain bus voltages and per-line P/Q
flows, providing the operating point for AC certificate construction.

**Implementation file:**
`src/stability_radius/base_point/pypsa_pf.py`, function
`solve_ac_pf_base_point_from_pandapower` (line 547)

#### 3-Attempt Retry Cascade

The AC PF solver employs a robust 3-attempt cascade to handle difficult networks:

```
Attempt 1 (primary):
    init = configured (flat/dc)
    enforce_q_lims = True
    distributed_slack = as configured

Attempt 2 (alt_init):
    init = opposite of primary (dc/flat)
    enforce_q_lims = True
    (triggered only if attempt 1 fails)

Attempt 3 (relaxed):
    init = flat
    enforce_q_lims = False
    distributed_slack = False
    (triggered only if attempts 1 and 2 fail)
```

#### Solver Backends

- **`solver="pandapower"`**: Direct `pandapower.runpp()` call
- **`solver="pypsa"`**: Converts to PyPSA network, runs `n.pf()`

#### Key Outputs (PyPSAAPFResult)

- `v_mag_pu`: Bus voltage magnitudes (per-unit)
- `v_ang_rad`: Bus voltage angles (radians)
- `line_p0_mw`, `line_q0_mvar`: P/Q at from-end per line
- `line_p1_mw`, `line_q1_mvar`: P/Q at to-end per line
- `bus_p_mw`, `bus_q_mvar`: Net bus injections

### 7.4 AC Feasible Power Flow (AC FPF) via pandapower.runopp()

**Name:** `solve_ac_fpf`

**Purpose:** Find the closest AC-feasible operating point to an initial dispatch
guess by solving a quadratic feasibility minimization problem.

**Implementation file:**
`src/stability_radius/base_point/pandapower_opp.py`, function `solve_ac_fpf` (line 330)

#### Objective Function

```
min  Sum_i (P_{g,i} - P_{g,i}^0)^2
```

Implemented via pandapower polynomial cost:
```
cp2 = 1,  cp1 = -2*P0,  cp0 = P0^2
```

#### Constraints

- AC power flow equations (equality)
- Generator P/Q bounds (inequality)
- Bus voltage magnitude bounds: `V_min <= V <= V_max`
- Line thermal limits: `|S| <= S_max`

#### Retry Chain

```
Attempt 1: configured bounds [vm_min, vm_max], init=dc
Attempt 2: relaxed V bounds [0.85, 1.15], init=flat
Attempt 3: further relaxed [0.80, 1.20], init=flat
```

#### Post-OPP PF Validation

After OPP converges, the dispatch is applied back to the network and
`pandapower.runpp()` is re-run. This ensures the base point used by the
AC certificate matches exactly what Newton-Raphson produces, avoiding
solver-dependent discrepancies between PIPS and Newton-Raphson.

#### Solver Backend

Uses PYPOWER's primal-dual interior point solver (PIPS) via
`pandapower.runopp()`. No external NLP solver (IPOPT, Pyomo) is required.

---

## 8. Monte Carlo Verification

**Name:** `run_monte_carlo_verification`

**Purpose:** Empirically verify the stability radius certificate by sampling
random injection perturbations, recomputing flows, and checking for overloads.
Supports both DC and AC modes.

**Implementation file:**
`src/stability_radius/verification/monte_carlo.py`, function
`run_monte_carlo_verification`

### Algorithm (DC Mode)

```
PROCEDURE mc_verification_dc(results, net, n_samples, seed):
    1. LOAD CERTIFICATE
       r_star <- min radius from results
       base_flows <- per-line flows from results

    2. BUILD DC OPERATOR
       op <- build_dc_operator(net, slack_bus)

    3. GENERATE RANDOM PERTURBATIONS
       FOR each sample s in 1..n_samples:
           delta_p <- Gaussian(0, sigma^2) for each bus
           # Enforce power balance: delta_p <- delta_p - mean(delta_p)
           Scale to ||delta_p||_2 = r_star (boundary of certified ball)

    4. COMPUTE PERTURBED FLOWS
       delta_f <- op.flows_from_delta_injections(delta_p)
       f_perturbed <- base_flows + delta_f

    5. CHECK FOR OVERLOADS
       FOR each line l:
           IF |f_perturbed[l]| > limit[l]:
               violation_count[l] += 1

    6. COMPUTE EMPIRICAL PROBABILITIES
       empirical_prob <- violation_count / n_samples

    7. COMPARE WITH CERTIFICATE
       soundness: PASS if no violations within certified ball
       probabilistic: compare empirical rate with analytic bound
```

### Algorithm (AC Mode)

```
PROCEDURE mc_verification_ac(results, net, n_samples, seed):
    1. LOAD CERTIFICATE AND BASE POINT
       Restore AC PF base point (Vm, Va, solver, lossless)

    2. FOR each sample:
       a. Generate Gaussian perturbations (delta_P, delta_Q)
       b. Apply perturbations as sgen at each bus
       c. Run pandapower.runpp() to obtain perturbed flows
       d. Compute |S| at each line end
       e. Check against thermal limits

    3. Handle PF failures (count, skip)

    4. REPORT
       Per-line overload fractions
       Soundness check (within certified ball)
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `n_samples` | Number of random perturbations |
| `seed` | Random seed for reproducibility |
| `ac_sigma_p_mw` | Per-bus active power std dev for sampling |
| `ac_sigma_q_mvar` | Per-bus reactive power std dev for sampling |
| `track_per_line_overloads` | Enable per-line overload counting |

### Computational Complexity

- DC mode: O(n_samples * n * m) -- dominated by LU back-substitutions
- AC mode: O(n_samples * PF_cost) -- much more expensive due to full AC PF per sample

---

## 9. Certificate Verification

The project provides two complementary verification approaches.

### 9.1 Worst-Case Perturbation Verification

**Name:** `verify_worst_case`

**Purpose:** Verify a single worst-case perturbation by running a full nonlinear
AC power flow and comparing the predicted apparent power against the actual.

**Implementation file:**
`src/stability_radius/verification/verify_worst_case.py`, function
`verify_worst_case` (line 142)

#### Algorithm

```
PROCEDURE verify_worst_case(net, line_id, h_vec, radius, s0, ...):
    1. CONSTRUCT WORST-CASE PERTURBATION
       IF delta_u not provided:
           h <- h_vec (copy)
           IF balance: project P/Q blocks (mean subtraction)
           direction <- h / ||h||_2
           delta_u <- direction * radius * scale

    2. PREDICT APPARENT POWER (linear model)
       S_predicted <- s0 + h^T * delta_u

    3. APPLY PERTURBATION TO NETWORK
       nn <- deep_copy(net)
       IF lossless: apply_lossless_policy(nn)
       FOR each bus:
           create_sgen(nn, bus, p_mw=dp[bus], q_mvar=dq[bus])

    4. RUN NONLINEAR AC PF
       pandapower.runpp(nn)

    5. EXTRACT ACTUAL APPARENT POWER
       S_from <- sqrt(p_from^2 + q_from^2)
       S_to   <- sqrt(p_to^2 + q_to^2)
       S_actual <- S_binding_end (or max)

    6. COMPUTE METRICS
       violated <- S_actual > limit_mva
       relative_error <- |S_predicted - S_actual| / S_actual

    7. RETURN WorstCaseVerificationResult
```

### 9.2 Violation Scale Search (Binary Search)

**Name:** `find_violation_scale`

**Purpose:** Find the actual perturbation scale at which nonlinear PF violates
the thermal limit, quantifying the conservatism of the linear certificate.

**Implementation file:**
`src/stability_radius/verification/verify_worst_case.py`, function
`find_violation_scale` (line 372)

#### Algorithm

```
PROCEDURE find_violation_scale(net, line_id, h_vec, radius, ...):
    Phase 1: FIND UPPER BOUND
        Start at scale=0 (no violation expected)
        Try scale=1, 2, 4, ... up to scale_max
        Stop when violation found or PF diverges

    Phase 2: BINARY SEARCH
        lo <- last non-violating scale
        hi <- first violating scale
        WHILE hi - lo > tol AND iterations < max_iter:
            mid <- (lo + hi) / 2
            Run PF at scale=mid
            IF violated: hi <- mid
            ELSE: lo <- mid

    RETURN ViolationScaleSearchResult
        conservatism_ratio = actual_scale / 1.0
        (>1 means conservative, <1 means optimistic)
```

### 9.3 Certificate Interpretation

**Name:** `interpret_certificate`

**Purpose:** Provide typed semantic labels for certificate quality.

**Implementation file:**
`src/stability_radius/verification/verify_certificate.py`, function
`interpret_certificate_components` (line 75)

#### Classification Logic

| Condition | Soundness | Usefulness |
|-----------|-----------|------------|
| Base infeasible | `unknown` | `n/a` |
| r* not finite | `unknown` | `n/a` |
| r* < 0 | `unknown` | `n/a` |
| r* = 0 | `trivial_true` | `zero_radius` |
| r* > 0, MC PASS | `sound` | `nonzero_radius` |
| r* > 0, MC FAIL | `unsound` | `nonzero_radius` |
| r* > 0, no MC | `unknown` | `nonzero_radius` |

---

## 10. Metrics Analysis Pipeline

**Name:** `main` (CLI entry point)

**Purpose:** Comparative evaluation of stability radii against baseline
robustness metrics, producing correlation analyses, precision-at-k rankings,
and visualizations.

**Implementation file:**
`src/stability_radius/metrics_analysis.py` (line 386)

### Pipeline Steps

```
Step 1: COMPUTE ALL AC RADII
    results <- compute_results_for_case(
        compute_ac=True,
        ac_extensions={sigma, metric}
    )
    Save to results.json

Step 2: MONTE CARLO WITH PER-LINE TRACKING
    vr <- run_monte_carlo_verification(
        mode="ac",
        track_per_line_overloads=True
    )
    Extract per-line overload fractions

Step 3: COMPUTE BASELINE METRICS
    FOR each line:
        loading_ratio <- s0 / s_limit
        headroom_mva <- s_limit - s0
        cheb_prob_upper <- Cantelli bound

Step 4: ANALYSIS AND VISUALIZATION
    a. Build unified DataFrame (one row per line, all metrics)
    b. Compute Spearman rank correlations vs empirical overload probability
    c. Compute precision-at-k (top-k most-dangerous lines)
    d. Generate scatter plots (each metric vs empirical probability)
    e. Generate Spearman bar chart
    f. Generate radius distribution histograms
    g. Save CSV outputs
```

### Metrics Compared

| Metric | Source | Interpretation |
|--------|--------|----------------|
| `radius_ac_l2` | AC L2 certificate | Smaller = more vulnerable |
| `radius_ac_sigma` | AC sigma certificate | Smaller = more vulnerable |
| `radius_ac_metric` | AC metric certificate | Smaller = more vulnerable |
| `loading_ratio` | `s0 / s_limit` | Larger = more loaded |
| `headroom_mva` | `s_limit - s0` | Smaller = more vulnerable |
| `cheb_prob_upper` | Cantelli bound | Larger = higher overload risk |
| `overload_probability_ac` | Gaussian Q-function | Larger = higher overload risk |

### Spearman Correlation Sign Convention

For "lower-is-more-dangerous" metrics (radii, headroom), the sign is negated
so that a positive Spearman rho consistently means "correctly identifies
dangerous lines."

### CLI Usage

```bash
python -m stability_radius.metrics_analysis \
    --input data/input/pglib_opf_case30_ieee.m \
    --slack-bus 0 \
    --sigma-p 1.0 --sigma-q 1.0 \
    --mc-samples 10000 \
    --output-dir case30
```

---

## 11. N-1 Contingency Radius

**Name:** `compute_nminus1_l2_radius`

**Purpose:** Compute effective N-1 L2 radii that account for single-line outage
contingencies using LODF (Line Outage Distribution Factor) approximations.

**Implementation file:**
`src/stability_radius/radii/nminus1.py`, function `compute_nminus1_l2_radius`
(line 250)

### Supporting Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `ptdf_for_line_transfers` | line 23 | Compute PTDF matrix `H @ E^T` |
| `lodf_from_ptdf` | line 44 | Derive LODF from PTDF with islanding handling |
| `incidence_from_pandapower_net` | line 101 | Build oriented incidence matrix E |
| `effective_nminus1_l2_radii` | line 133 | Core N-1 radius computation |

### Algorithm

```
PROCEDURE compute_nminus1_l2_radius(net, H_full, ...):
    1. BUILD INCIDENCE AND PTDF
       E <- oriented incidence (m x n), +1 at from, -1 at to
       PTDF <- H_full @ E^T      # (m x m)

    2. COMPUTE LODF
       FOR each contingency k:
           denom_k <- 1 - PTDF[k, k]
           IF |denom_k| < tol:   # islanding contingency
               LODF[:, k] <- NaN  (if islanding="skip")
               OR raise error     (if islanding="raise")
           ELSE:
               LODF[m, k] <- PTDF[m, k] / denom_k   for m != k
           LODF[k, k] <- -1

    3. COMPUTE N-1 RADII
       Precompute base norms and sums for balanced projection:
           g_norm2 <- sum(G^2, axis=1)
           g_sum   <- sum(G, axis=1)
           g_proj_norm2 <- g_norm2 - g_sum^2 / n_bus

       FOR each contingency k (column of LODF):
           IF LODF[:, k] contains NaN: skip (islanding)

           # Post-contingency flows:
           f_post <- f + LODF[:, k] * f_k
           f_post[k] <- 0  (outaged line has zero flow)

           # Post-contingency margins:
           margin_post <- max(c - |f_post|, 0)

           IF update_sensitivities:
               # Updated sensitivity: g_m^(k) = g_m + LODF[m,k] * g_k
               # Compute ||Proj(g_m + alpha_m * g_k)||_2 via:
               norm2_post <- g_norm2 + 2*alpha*dots + alpha^2*g_norm2[k]
               sum_post <- g_sum + alpha * g_sum[k]
               proj_norm2_post <- norm2_post - sum_post^2 / n_bus
               denom <- sqrt(max(proj_norm2_post, 0))
           ELSE:
               denom <- sqrt(g_proj_norm2)   # reuse base norms

           radii_k <- margin_post / denom
           radii_k[k] <- +inf   # skip outaged line

           # Track minimum across contingencies:
           best[m] <- min(best[m], radii_k[m])
           argmin[m] <- k where minimum achieved

    4. RETURN results with per-line N-1 radius and worst contingency
```

### LODF Mathematical Definition

The Line Outage Distribution Factor describes flow redistribution when line k
is removed:

```
LODF[m, k] = PTDF[m, k] / (1 - PTDF[k, k])
```

Post-contingency flow on line m when line k is outaged:

```
f_m^(k) = f_m + LODF[m, k] * f_k
```

### Sensitivity Update (Woodbury)

When `update_sensitivities=True`, the post-contingency sensitivity vector is:

```
g_m^(k) = g_m + LODF[m, k] * g_k
```

The projected norm is computed efficiently using precomputed dot products:

```
||Proj(g_m^(k))||^2 = ||g_m + alpha*g_k||^2 - (sum(g_m + alpha*g_k))^2 / n
```

### Islanding Handling

When `1 - PTDF[k,k]` is near zero, removing line k creates an island
(disconnected subnetwork). Two modes:

- `islanding="skip"`: Set LODF column to NaN, exclude from min computation
- `islanding="raise"`: Raise ValueError immediately

### Computational Complexity

| Step | Complexity |
|------|-----------|
| PTDF computation | O(m^2 * n) |
| LODF computation | O(m^2) |
| N-1 radius search | O(m^2 * n) with sensitivity updates, O(m^2) without |

### Design Rationale

- **LODF approximation:** Avoids rebuilding and re-factorizing the B matrix
  for each of the m possible contingencies, reducing complexity from
  O(m * n^{1.5}) to O(m^2).
- **Balanced norm consistency:** The same balanced projection
  `||Proj(g)||_2 = ||g - mean(g)*1||_2` is used as in the base N-0 certificate,
  ensuring slack-bus invariance.
- **Vectorized contingency sweep:** The inner loop over monitored lines is
  fully vectorized with NumPy, avoiding Python-level iteration over the
  m x m matrix.

---

## Cross-Reference: Key Data Types

| Type | Module | Purpose |
|------|--------|---------|
| `DCOperator` | `dc/dc_model.py` | DC linear operator with LU factorization |
| `ACOperator` | `ac/ac_model.py` | AC Jacobian operator with LU factorization |
| `LineBaseQuantities` | `radii/common.py` | Per-line base flows, limits, margins |
| `PyPSAOPFResult` | `base_point/pypsa_opf.py` | DC OPF solution |
| `PyPSAAPFResult` | `base_point/pypsa_pf.py` | AC PF solution (voltages + line flows) |
| `ACFPFConfig` | `base_point/pandapower_opp.py` | AC FPF solver configuration |
| `L2RadiusCertificate` | `radii/core_l2.py` | Pure L2 certificate (no pandapower deps) |
| `LODFResult` | `radii/nminus1.py` | LODF matrix with islanding metadata |
| `WorstCaseVerificationResult` | `verification/verify_worst_case.py` | Single perturbation verification |
| `ViolationScaleSearchResult` | `verification/verify_worst_case.py` | Binary search for violation scale |
| `CertificateInterpretation` | `verification/verify_certificate.py` | Soundness/usefulness labels |

---

## Cross-Reference: Units Contract

All algorithms in this project adhere to a consistent units convention:

| Quantity | Unit | Notes |
|----------|------|-------|
| Active power (P, flow, injection) | MW | |
| Reactive power (Q) | MVAr | |
| Apparent power (S) | MVA | |
| Voltage magnitude | per-unit (p.u.) | |
| Voltage angle | radians | |
| Bus nominal voltage | kV | |
| Impedance (series) | Ohm | Physical units |
| Branch susceptance (DC) | MW/rad | `b = V_kV^2 / X_ohm` |
| Admittance (AC) | per-unit | System base `sn_mva` |
| Stability radius (L2) | MW | Euclidean norm of injection perturbation |
| Stability radius (sigma) | dimensionless | Number of standard deviations |
| Stability radius (metric) | depends on M | Unit of M-norm ball |

---

## Cross-Reference: Configuration Constants

| Constant | Value | Location | Purpose |
|----------|-------|----------|---------|
| `_X_TOTAL_EPS` | `1e-12` | `dc/dc_model.py` | Zero-reactance threshold |
| `_TAP_RATIO_EPS` | `1e-12` | `dc/dc_model.py` | Zero-tap-step threshold |
| `_SHIFT_DEG_EPS` | `1e-9` | `dc/dc_model.py` | Zero-phase-shift threshold |
| `_EPS_Z_PU` | `1e-18` | `ac/ac_model.py` | Zero-impedance threshold (per-unit) |
| `_EPS_NORM` | `1e-12` | `radii/ac_l2.py` | Zero-sensitivity threshold |
| `_EPS_S0_MVA` | `1e-9` | `radii/ac_l2.py` | Zero apparent power threshold |
| `_EPS_SIGMA_FLOW` | `1e-15` | `radii/ac_sigma_radius.py` | Degenerate sigma threshold |
| `_EPS_DENOM` | `1e-12` | `radii/ac_metric_radius.py` | Metric denominator threshold |
| `_RATING_ZERO_EPS` | `1e-12` | `radii/common.py` | Zero-rating threshold |
