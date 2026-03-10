# Glossary

This glossary defines key terms, abbreviations, variable names, and domain-specific concepts used throughout the project and its documentation.

---

## A

**AC (Alternating Current)**
Power flow model that accounts for both active (P) and reactive (Q) power, voltage magnitudes, and phase angles. Nonlinear power flow equations.

**AC FPF (AC Feasibility Power Flow)**
AC power flow solved as an optimization problem (via `pandapower.runopp`) that minimizes the deviation from a target generator dispatch while respecting voltage and thermal constraints. Terminology used in this project for the AC OPP approach.

**AC L2 Radius**
The stability radius computed using the AC power flow Jacobian and apparent power flow constraints. Uses the L2 (Euclidean) norm for perturbations. See [mathematical_foundations.md](mathematical_foundations.md#6-ac-l2-stability-radius).

**AC PF (AC Power Flow)**
Full nonlinear power flow solution (Newton-Raphson) that computes voltages, angles, and power flows for a given dispatch.

**ACOperator**
The central data structure (`ac/ac_model.py`) encapsulating the sparse Ybus matrix, reduced Jacobian, and LU factorization for AC sensitivity computations.

**Adjoint Method**
Technique for computing sensitivities of a constraint with respect to all inputs simultaneously. Given J · dx = du, the adjoint sensitivity a = J^{-T} · b gives the gradient of the constraint with respect to injections.

**Apparent Power (S)**
The magnitude of complex power: |S| = √(P² + Q²), measured in MVA.

---

## B

**B Matrix (Bus Susceptance Matrix)**
The nodal susceptance matrix used in DC power flow: B = A^T · diag(b) · A, where A is the incidence matrix and b are branch susceptances.

**b (Branch Susceptance Coefficient)**
DC branch flow coefficient in MW/rad: b = V²_kV / X_ohm. Positive for inductive lines (standard).

**Balanced Disturbance**
A perturbation vector Δp satisfying 1^T · Δp = 0 (total injection change sums to zero), reflecting the physical power balance constraint.

**Base Dispatch / Base Point**
The operating point around which the stability radius is computed. Sources: case data, DC OPF, AC PF, or AC FPF.

**BasePointAC / BasePointDC**
Frozen dataclasses (`base_point/types.py`) storing the base operating point data for AC and DC models respectively.

**Binding End**
For AC analysis, the end of a line (from or to) that has the smaller radius (more restrictive constraint). Selected as min(r_from, r_to).

---

## C

**Cantelli Bound**
One-sided Chebyshev inequality: P(|S| > c) ≤ σ² / (σ² + margin²). A distribution-free upper bound on overload probability.

**Certificate**
A mathematical guarantee that all line constraints are satisfied within a specified perturbation ball. The L2 certificate guarantees: for ||Δp||₂ ≤ r*, all lines remain within thermal limits (under the linear model).

**Chunk Size**
Number of lines processed per batch in the adjoint solve. Controls peak memory usage. Configurable via `dc_chunk_size` / `ac_chunk_size`.

---

## D

**DC (Direct Current) Approximation**
Linear power flow model assuming: (1) lossless lines (r=0), (2) flat voltage profiles (|V|=1), (3) small angle differences. Flow is proportional to angle difference: f = b · (θ_from − θ_to).

**DCOperator**
The central data structure (`dc/dc_model.py`) encapsulating the sparse B matrix, LU factorization, incidence matrix, and sensitivity computation methods.

**Dual Norm**
The norm on the sensitivity space that determines the worst-case perturbation. For L2 perturbations, the dual is also L2. For metric-norm perturbations (||Δp||_M), the dual is ||·||_{M^{-1}}.

---

## E

**ext_grid**
Pandapower's external grid element, representing the slack bus connection. Provides voltage reference and absorbs power imbalance.

---

## F

**FPF (Feasibility Power Flow)**
See AC FPF.

---

## G

**g_l (Sensitivity Row Vector)**
Row l of the PTDF/sensitivity matrix H. Maps injection perturbations to flow change on line l: Δf_l = g_l^T · Δp.

---

## H

**H Matrix (PTDF Matrix)**
Power Transfer Distribution Factor matrix. Maps bus injection changes to line flow changes: Δf = H · Δp. Shape: (n_lines × n_buses).

**h-vector**
In the AC context, the adjoint sensitivity vector of apparent power |S| with respect to bus injection perturbations [ΔP; ΔQ]. Computed via J^{-T} · b where b encodes the constraint gradient.

**Headroom**
Thermal margin in MVA: headroom = S_limit − |S0|. A simple baseline robustness metric.

**HiGHS**
Open-source LP/MIP solver used for DC OPF via PyPSA/linopy.

---

## I

**Incidence Matrix (A)**
Oriented node-branch incidence matrix: A[l, from_bus] = +1, A[l, to_bus] = −1.

---

## J

**Jacobian (J)**
The AC power flow Jacobian matrix relating state changes [Δθ; ΔV] to injection changes [ΔP; ΔQ]. Reduced form excludes the slack bus and PV bus voltage variables.

---

## L

**L2 Radius**
Stability radius under the Euclidean (L2) norm: r_l = margin_l / ||g_l||₂. The global certificate is r* = min_l r_l.

**LineBaseQuantities**
Dataclass (`radii/common.py`) holding per-line base flows, limits, and margins.

**line_key**
Naming convention for result dict keys: `"line_{id}"` where id is the pandapower line index.

**Loading Ratio**
Baseline metric: loading_ratio = |S0| / S_limit. Ranges from 0 (unloaded) to 1 (at thermal limit).

**Lossless Policy**
Design choice to set resistance r=0 and disable shunt elements in the AC model, ensuring consistency between the certificate's Jacobian and the verification PF model.

---

## M

**M (Weight Matrix)**
Symmetric positive definite matrix defining a generalized norm ||Δp||_M = √(Δp^T M Δp). Used in metric radius computation.

**Margin**
The gap between the thermal limit and the base flow: margin = c − |f0| (DC, in MW) or margin = c − |S0| (AC, in MVA).

**MATPOWER**
Standard power system test case format (.m files). Used as the primary input format for this project.

**Metric Radius**
Stability radius under a generalized M-norm: r_l^(M) = margin_l / √(g_l^T M^{-1} g_l).

**Monte Carlo Verification**
Empirical validation of the analytic certificate by sampling random perturbations and checking for constraint violations.

**MVA (Megavolt-Ampere)**
Unit of apparent power. Used for AC thermal limits.

**MW (Megawatt)**
Unit of active power. Used for DC flows and bus injections.

---

## N

**N-1 Contingency**
Security criterion requiring the system to remain feasible after the loss of any single element (typically a transmission line).

**N-1 Radius**
L2 radius computed under each single-line contingency, reporting the worst binding constraint.

---

## O

**OPF (Optimal Power Flow)**
Optimization problem that determines the minimum-cost generator dispatch subject to network constraints.

**OPP (Optimal Power Point / pandapower.runopp)**
Pandapower's AC OPF solver, used for the AC FPF base dispatch.

**Overload Probability**
Probability that line flow exceeds the thermal limit: P(|S| > c). Computed analytically assuming Gaussian perturbations.

---

## P

**pandapower**
Python library for power system analysis. Used as the intermediate network representation and for AC power flow solutions.

**PDIPM**
Primal-Dual Interior Point Method. The solver algorithm used by pandapower.runopp() for AC OPF.

**PGLib-OPF**
Power Grid Library for benchmarking OPF algorithms. Provides standardized test cases.

**PQ Bus**
A bus where both active power (P) and reactive power (Q) injections are specified. Voltage is free.

**PTDF (Power Transfer Distribution Factor)**
Sensitivity matrix relating injection changes to flow changes. Equivalent to H matrix.

**PV Bus**
A bus where active power (P) and voltage magnitude (V) are specified. Reactive power is free (determined by the generator).

**PyPSA**
Python for Power System Analysis. Used for DC OPF formulation with HiGHS solver.

---

## Q

**Q-function**
Tail probability of the standard normal distribution: Q(x) = P(Z > x) = 0.5 · erfc(x/√2).

---

## R

**r* (r-star)**
Global stability radius certificate: r* = min_l r_l. The minimum per-line radius across all monitored lines.

**rateA**
MATPOWER field for line thermal rating in MVA. Primary thermal limit used in this project.

**Reduced System**
The power flow system after eliminating the slack bus. Has n−1 buses, n−1 equations.

---

## S

**S_limit / c**
Symmetric thermal limit for a line, in MVA (AC) or MW (DC).

**Sigma-Radius (r_σ)**
Stability radius in units of standard deviations: r_σ = margin / σ_flow, where σ_flow is the standard deviation of the flow perturbation under the injection covariance.

**σ_flow (sigma_flow)**
Standard deviation of the line flow under Gaussian bus injection perturbations: σ_flow = ||Σ^{1/2} h||₂.

**Slack Bus**
Reference bus with fixed voltage angle (θ=0). Absorbs power imbalance in the system.

**Soundness Check**
Verification that no violation is observed inside the certified ball. If a Monte Carlo sample within ||Δp|| ≤ r* causes a violation, the certificate is unsound.

**Spearman Rank Correlation**
Nonparametric measure of how well a metric's ranking of lines agrees with the empirical overload ranking from Monte Carlo.

---

## T

**Tap Ratio**
Transformer turns ratio. In the DC model: b_eff = b_raw / tap. In the AC model: complex tap a = tap · e^{jφ}.

**Thermal Limit**
Maximum permissible apparent power flow on a transmission line, determined by conductor heating.

---

## U

**UnitCommitment.jl**
Julia package for unit commitment optimization. The project uses its JSON format to import per-bus hourly demand data for computing realistic sigma arrays.

---

## V

**VerificationResult**
Dataclass (`verification/types.py`) containing all verification outcomes: base point check, radius check, soundness check, probabilistic check.

---

## W

**Woodbury Update**
Matrix identity for rank-k updates: (A + UCV)^{-1}. Used in N-1 analysis to efficiently update the sensitivity matrix after a line outage.

**Worst-Case Perturbation**
The injection perturbation that maximizes the flow on a specific line within the certified ball. For L2 radius: Δp* = r · g / ||g||. For sigma-radius: Δu* = r_σ · (Σ h) / σ_flow.

---

## Y

**Ybus**
Bus admittance matrix in per-unit. Encodes the network topology and impedances. Used in AC power flow: I = Ybus · V.

---

## Abbreviations

| Abbreviation | Full Name |
|-------------|-----------|
| AC | Alternating Current |
| DC | Direct Current |
| FPF | Feasibility Power Flow |
| GOC | Grid Optimization Competition |
| HiGHS | High-performance software for linear optimization |
| IEEE | Institute of Electrical and Electronics Engineers |
| LU | Lower-Upper (matrix decomposition) |
| MC | Monte Carlo |
| MVA | Megavolt-Ampere |
| MW | Megawatt |
| OPF | Optimal Power Flow |
| OPP | Optimal Power Point (pandapower) |
| PDIPM | Primal-Dual Interior Point Method |
| PF | Power Flow |
| PGLib | Power Grid Library |
| PQ | Active-Reactive (bus type) |
| PTDF | Power Transfer Distribution Factor |
| PV | Active-Voltage (bus type) |
| RHS | Right-Hand Side |
| RNG | Random Number Generator |
| SPD | Symmetric Positive Definite |
