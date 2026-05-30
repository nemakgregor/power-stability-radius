# Reviewer Simulation for `paper/manuscript.tex`

Date: 2026-05-29.

This file is a pre-submission risk review, not part of the manuscript.

## Reviewer A: Novelty

### Strengths

- The paper is framed around a clear methodological contribution: balanced stability-radius certificates for DC and AC thermal security.
- The AC adjoint formulation, binding line-end handling, and metric/sigma unification give more novelty than a pure application study.
- The introduction states a precise gap: fast post-dispatch metrics do not certify balanced multidimensional perturbation sets, while robust/chance OPF does not directly evaluate arbitrary existing operating points.

### Weaknesses

- The current draft must avoid implying that all pieces are globally novel in isolation. PTDF, LODF, chance constraints, robust uncertainty sets, and adjoints are established.
- The N-1 dispatch demo can distract from the main contribution unless it is positioned as an application, not the central novelty.

### Missing Citations

- Additional final citations should be added for SCOPF and contingency screening.
- If the final submission emphasizes dispatch optimization, cite radius/robust-margin or reserve-margin optimization papers in power systems.

### Rejection Risks

- Risk: reviewer says the method is "just sensitivity divided by headroom."
- Mitigation already in draft: prove balanced slack-invariant projection, AC line-end adjoint formulation, metric/sigma equivalence, and benchmark against heuristic headroom/sensitivity metrics.
- Remaining mitigation: add a compact synthetic example showing two lines with identical loading but different radii and different Monte Carlo overload rates.

## Reviewer B: Methodology

### Strengths

- Variables, sets, margins, perturbation spaces, and constraints are defined explicitly.
- Theorem 1 gives the core certificate guarantee.
- Proposition 1 addresses slack-reference invariance, which is critical for PTDF-based certificates.
- Proposition 2 gives a coherent link between metric and sigma radii.

### Weaknesses

- The AC certificate is local and depends on a fixed PV/PQ classification. Reviewers will ask how often nonlinear replay violates the linear certificate.
- The draft currently summarizes the algorithm but does not yet provide a full derivation of the AC apparent-power gradient.
- The N-1 section is DC-centric; if kept, the final paper should clearly distinguish DC LODF screening from AC nonlinear contingency verification.

### Missing Experiments

- Boundary replay at alpha values below, at, and above 1.0 for the bottleneck line across several systems.
- Ablation of balanced projection versus unbalanced projection.
- Ablation of binding-end selection versus from-end-only checks.
- AC sigma projection ablation: variance-weighted balance versus ordinary mean subtraction.

### Rejection Risks

- Risk: local AC linearization may be seen as unreliable.
- Mitigation already in draft: nonlinear replay and PF-failure reporting are required.
- Remaining mitigation: include a table of crossing-alpha statistics and failure counts across small, medium, large, and stressed cases.

## Reviewer C: Experiments

### Strengths

- The draft defines RQs before presenting numbers.
- Baselines include industry metrics, probabilistic bounds, optimization baselines, and ablations.
- Preliminary results already show meaningful behavior: AC/DC divergence, stronger case118 rank correlation for radius metrics, and a cost/security tradeoff in the N-1 demo.

### Weaknesses

- Existing results are from multiple artifact folders and configurations. They are useful for drafting but not submission-grade.
- The preliminary PGLib sweep has timeouts, AC PF failures, and negative radii under mixed base-point policies.
- Statistical rigor is not yet complete: repeated runs, confidence intervals, and significance tests are specified but not yet executed for all experiments.

### Missing Experiments

- Locked full PGLib sweep with one documented configuration and explicit failure accounting.
- Scalability study with repeated timings and memory measurements.
- Monte Carlo with at least three seeds per selected case.
- Sensitivity to sigma scaling and uncertainty heterogeneity.
- Comparison to SCOPF or N-1 screening on more than one case if the dispatch use case remains in the results section.

### Rejection Risks

- Risk: reviewers reject because the empirical evidence is fragmented.
- Mitigation already in draft: current results are labeled preliminary, and the required submission protocol is explicit.
- Remaining mitigation: regenerate all tables from one scripted pipeline and include artifact checksums.

## Revision Actions Already Reflected in the Draft

- Framed the contribution as a certificate model and adjoint algorithm, not a software package.
- Separated demonstrated preliminary evidence from the final benchmark protocol.
- Added theoretical statements and proofs.
- Added mandatory ablations, baselines, statistical reporting, and failure accounting to the experimental design.
- Added limitations on AC local validity, PF convergence, base infeasibility, islanding, and Monte Carlo cost.

## Blocking Items Before Q1 Submission

- Run and freeze the final experiments from a single commit.
- Replace preliminary tables with final regenerated tables and confidence intervals.
- Add a full SCOPF/security-constrained literature paragraph.
- Add exact software versions and hardware details.
- Decide whether the target contribution is only the certificate/evaluation framework or also a radius-aware dispatch model.
