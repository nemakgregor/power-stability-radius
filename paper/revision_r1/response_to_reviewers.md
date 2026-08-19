# Response to Reviewers — Access-2026-34750 (Resubmission)

**Manuscript:** Thermal-Security Robustness Radii for Power Systems (revised title: *Adjoint-Based Thermal-Security Robustness Radii for AC Power Systems: A Post-Dispatch Screening and Diagnostic Certificate Under Response-Constrained Injection Uncertainty*)

We thank both reviewers. Reviewer 1 recommended acceptance. Reviewer 2 provided twelve substantive requirements; we address each below. The single most important outcome of this revision: **the two central technical concerns (the finite-difference discrepancy and the sigma miscalibration) shared one root cause — two implementation defects, which we found, fixed, regression-tested, and re-validated to the power-flow-tolerance floor.** All revision experiments regenerate from committed scripts and artifacts in the public repository (branch `revision-r1-screening`).

Formatting of each item: **(a)** reviewer's concern → **(b)** our response → **(c)** action taken.

---

## Item 1 — Rewrite novelty around the steady-state security-distance literature

**(b)** We agree. The concept of distance from an operating point to a security boundary is established (Chen et al. 2013, 2015a, 2015b), and our contribution is a constrained-geometry, AC-adjoint member of that family, not the first distance construction.

**(c)** The Introduction and Literature Review are rewritten around SSSD. A dedicated side-by-side table (Table "Positioning", covering uncertainty space, metric, constraints, network model, contingency treatment, nonlinear guarantee, worst-direction recovery, and complexity) compares SSSD, convex inner approximations (Nguyen et al. 2019), unified robustness diagnostics (Anton & Ilić 2026, now cited in its journal version), and this work. The theorem is now explicitly presented as "the standard support-function argument … a unifying engineering specialization rather than a new theoretical result."

## Item 2 — Resolve the finite-difference discrepancy

**(b)** Resolved, with a root cause rather than a caveat. The non-convergent 1–4% plateau had two causes, both implementation defects, not model properties:
1. **Slack-index inconsistency.** The AC operator resolved a positional `slack_bus: 0` to bus position 0, while the sensitivity expansion resolved it (ext-grid-aware) to the true slack position. On any network whose ext_grid is not the first sorted bus (case118: bus 69; case200: bus 189), every `h_P` entry below the slack position was shifted by one bus. This corrupted FD diagnostics, sigma radii, and worst-case directions — exactly the two quantities the review flagged.
2. **Unmodeled reactive-limit active set.** The base PF enforces Q-limits (case118: 29 saturated generators); buses whose every controlling generator is limit-pinned are effectively PQ in the converged solution, while the operator classified them as PV.

**(c)** Both fixed (slack resolution unified and returned by the certificate itself; fully saturated PV buses linearized as PQ), regression-tested (a dedicated test builds a network with the ext_grid deliberately not at the first bus; FD tolerances tightened from 0.05 to 0.01; the certificate's admittance model is asserted equal to the verification solver's internal Ybus), and re-verified: centered FD along the balanced worst-case direction over eps ∈ {1, 0.1, 0.01, 0.001} MVA now decreases as O(eps²) — case118 median relative error 4.1e-7 → 3.1e-11, case200 6.6e-9 → 7.5e-11 — down to the PF-tolerance floor (new adjoint-validation table). The adjoint residual ‖JᵀH−B‖∞/max(1,‖B‖∞) is now computed inside every certificate run (≤2e-15 in all reported runs). The branch-flow gradients are stated in the manuscript with all sign and base-MVA conventions.

## Item 3 — Separate "affine-model certificate", "nonlinear robustness indicator", "probability estimate"

**(c)** The revision is repositioned end-to-end as a **post-dispatch screening and diagnostic certificate**. A single scope statement in the Introduction is enforced through abstract, results, discussion, and conclusions. Empirical nonlinear safety at the predicted radius is now quantified (Item 4), and probabilistic language is confined to the calibrated regime (Item 5).

## Item 4 — Expand nonlinear replay

**(c)** New multi-scale replay experiment: for each case, the five tightest lines are replayed along their certified worst directions at α ∈ {0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5}·r, recording thermal violations of ALL monitored lines, voltage-band violations, Q-limit activations, and PF convergence. Result: the interpolated nonlinear crossing sits at **0.90–1.00 of the affine radius** (median 0.95–0.999 per case) — the affine radius is mildly optimistic, by at most 10% and typically under 5%, and the paper now states this quantitatively with a de-rating recommendation. All replays converged; no voltage violations occurred at α ≤ 1.

## Item 5 — Investigate the sigma miscalibration

**(b)** The 11.026 vs 3.950 MVA discrepancy (ratio 2.8) was the slack-index defect of Item 2 acting on `line_85`'s sensitivity — not conditioning-vs-response confusion, not covariance error, not nonlinearity.

**(c)** With corrected sensitivities, line-wise calibration across the ten highest-variance ends of case118 gives **analytical/empirical sd ratios with median 1.004, range [0.989, 1.017]** (3000 nonlinear MC replays, zero PF failures), and tightened-limit exceedance probabilities inside or near the Wilson 95% intervals (new calibration table). The conditioning-vs-response distinction the reviewer raised is now stated explicitly in Methods, and each experiment declares which mechanism it uses. The manuscript still refrains from calling β a system-level reliability index.

## Item 6 — Strengthen the admissible-response model

**(c)** New experiment: uncertainty restricted to load buses, generators responding via headroom-proportional participation factors (Σα=1). For the five tightest 118-bus lines: radii shift −1% to +21% vs the blockwise-balanced convention; **every worst-case perturbation respects all generator [p_min, p_max] limits**; nonlinear replay with the explicit generator redispatch at the certified radius reaches |S|/limit = 1.001–1.015 — realizable and tight. The Discussion covers inequality-constrained sets (box/polyhedral → SOCP support problem) and the closed-form-vs-realism trade-off.

## Item 7 — Complete the treatment of zero-flow line ends

**(c)** Implemented. Zero-flow ends now receive the exact first-order operator-norm radius: two adjoint solves give the 2×d map [∇P; ∇Q]; after balanced projection, r = c_ℓ / σ_max, a 2×2 eigenvalue problem per end. On case2000_goc all 17 zero-flow lines are certified and the **all-constraint system radius is now defined** (previously the case was only partially certified). The detection threshold is now scale-aware (max(1e-9, 1e-9·c_ℓ) MVA) and the end-count is shown to be insensitive to the threshold over eight orders of magnitude. A tightness test (symmetric-diamond network) verifies the operator-norm bound is achieved within 5% along the top singular direction.

## Items 8 & 9 — Dispatch evaluation and algorithm/formulation alignment

**(b)** We accept the criticism in full and have drawn the conclusion the review itself suggests: a defensible dispatch study requires matched-cost baselines and a recognized preventive/corrective SCOPF comparator across several lossy systems — a separate paper.

**(c)** The dispatch experiment, the tightened-limit heuristic, the SCOPF proxy, and all associated claims are **removed from this manuscript**. The paper is now strictly a screening/diagnostic contribution; the DC LODF layer is retained only as a screening extension with an explicit "not a dispatch method" statement. Radius-aware dispatch is listed as future work with the evaluation protocol the reviewer specified.

## Item 10 — Strengthen statistical analysis

**(c)** The ranking study now uses **scenario-level bootstrap** (resampling Monte Carlo scenarios, preserving cross-line correlation; 10,000 converged nonlinear scenarios, 2000 bootstrap replicates), reports **paired** bootstrap CIs for the Spearman differences (radius vs loading, radius vs headroom — both excluding zero), and adds precision@k / recall@k and the explicit false-negative line lists at k ∈ {3,5,10}. The line-bootstrap of the original submission is dropped.

## Item 11 — Revise the scalability claims

**(c)** The timing table now separates AC PF / operator build (assembly+LU) / certificate (adjoint+norms) with mean±std over 7 repetitions and **measured peak resident memory**, plus stored-h array size as a separate column. Hardware for the revision runs is declared, and the manuscript no longer claims broad large-case scalability: the failure accounting stays, and the case2000 study explicitly reports that the lossless runpp base point does not converge and an AC-FPF base point is used instead.

## Item 12 — Presentation

**(c)** Added: a notation-and-units table; an explicit units column distinguishing MW radii, per-unit/mixed radii, and dimensionless sigma radii; running header fixed to the single author; repository to be archived at a fixed release with DOI at resubmission, with the exact commit and script named for every table.

## References

Added: Chen et al. 2013 (IET GTD 7(3)), Chen et al. 2015 (IET GTD 9(15)), Chen et al. 2015 (Proc. CSEE 35(3)), Nguyen et al. 2019 (IEEE TPWRS 34(1)). Updated: Anton & Ilić to the journal version (IJEPES 178:111979, 2026). Now cited in the text: PGLib-OPF (arXiv:1908.02788), UnitCommitment.jl (Zenodo 10.5281/zenodo.4269874). Removed as peripheral (per item 5a): the four metaheuristic-optimization references and the generic robust-optimization-propagation reference.

---

### Reproducibility of this response

Every number above regenerates from `experiments/revision_r1/scripts/exp{1..8}_*.py` at the stated commit; JSON artifacts are committed under `experiments/revision_r1/results/`. The two defects and their fixes are covered by new regression tests (`tests/test_slack_extgrid_regression.py`, `tests/test_ybus_matches_pandapower.py`, `tests/test_zero_flow_operator_norm.py`) and the tightened `tests/test_h_vector_fd.py` tolerances.
