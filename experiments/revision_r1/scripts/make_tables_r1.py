"""Generate the revised manuscript tables from the revision-r1 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
OUT = Path(__file__).resolve().parents[3] / "paper" / "revision_r1" / "tables_r1"
OUT.mkdir(parents=True, exist_ok=True)


def esc(x):
    return str(x).replace("_", "\\_")


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def write(name: str, text: str) -> None:
    (OUT / name).write_text(text)
    print(f"wrote {OUT / name}")


def sci(x, digits=2):
    if x is None:
        return "--"
    return f"\\num{{{x:.{digits}e}}}".replace("e-0", "e-").replace("e+0", "e+")


def fixed(x, d=3):
    return "--" if x is None else f"{x:.{d}f}"


# ---------------- adjoint validation (exp1 + exp5) ----------------
d1, d5 = load("exp1_fd_convergence.json"), load("exp5_timing_breakdown.json")
if d1:
    t5 = {c["case"]: c for c in (d5 or {"cases": []})["cases"] if "failed" not in c}
    rows = []
    for c in d1["cases"]:
        a = c["aggregate_by_eps"]
        t = t5.get(c["case"], {})
        rows.append(
            f"\\texttt{{{esc(c['case'])}}} & {c['n_ends_checked']} & "
            f"{sci(a['1.0']['median'])} & {sci(a['0.01']['median'])} & "
            f"{sci(a['0.001']['median'])} & {sci(a['0.001']['max'])} & "
            f"{sci(c['adjoint_residual_max'])} & "
            f"{fixed(t.get('certificate_adjoint_plus_norms', {}).get('mean_s'), 4)} \\\\"
        )
    body = "\n".join(rows)
    write(
        "table_adjoint_validation.tex",
        r"""\begin{table}[t]
\centering
\caption{Adjoint verification after the implementation corrections: centered
finite differences along the balanced worst-case direction of the ten
tightest differentiable line ends per case.  The relative error now decreases
as $O(\varepsilon^2)$ down to the power-flow-tolerance floor, in contrast to
the non-convergent $10^{-2}$-level plateau of the original submission, which
was traced to a slack-index defect and an unmodeled reactive-limit active
set.  Certificate time is mean of 7 repetitions (adjoint + norms, excluding
AC PF).\label{tab:adjoint-validation}}
\scriptsize
\begin{tabularx}{\textwidth}{@{}Xrrrrrrr@{}}
\toprule
Case & Ends & Med.\ rel.\ ($\varepsilon{=}1$) & Med.\ rel.\ ($\varepsilon{=}10^{-2}$) & Med.\ rel.\ ($\varepsilon{=}10^{-3}$) & Max rel.\ ($\varepsilon{=}10^{-3}$) & Adj.\ resid. & Cert.\ (s) \\
\midrule
"""
        + body
        + "\n"
        + r"""\bottomrule
\end{tabularx}
\end{table}
""",
    )

# ---------------- sigma calibration (exp2) ----------------
d2 = load("exp2_sigma_calibration.json")
if d2:
    rows = []
    for r in d2["lines"]:
        rows.append(
            f"\\texttt{{line\\_{r['line']}}} & {r['end']} & {fixed(r['s0_mva'], 2)} & "
            f"{fixed(r['sigma_flow_analytical_mva'], 3)} & "
            f"{fixed(r['sigma_flow_empirical_mva'], 3)} & "
            f"{fixed(r['ratio_analytical_over_empirical'], 3)} & "
            f"{fixed(r['predicted_exceed_prob'], 4)} & {fixed(r['empirical_exceed_prob'], 4)} & "
            f"[{fixed(r['wilson95'][0], 4)}, {fixed(r['wilson95'][1], 4)}] \\\\"
        )
    body = "\n".join(rows)
    rs = d2["ratio_summary"]
    write(
        "table_sigma_calibration.tex",
        r"""\begin{table}[t]
\centering
\caption{Line-wise sigma calibration on the IEEE 118-bus system
(load-proportional $\sigma_P$, $\sigma_Q=\sigma_P\tan(\arccos 0.9)$,
"""
        + f"{d2['n_mc']} nonlinear Monte Carlo replays, seed {d2['seed']}, 0 PF failures"
        + r""").
Ten highest-variance binding ends.  The analytical first-order standard
deviation now matches the empirical nonlinear standard deviation
(median ratio """
        + f"{rs['median']:.3f}, range [{rs['min']:.3f}, {rs['max']:.3f}]"
        + r""");
the tightened-limit exceedance probability at $s_0+2\hat\sigma$ is compared
with the empirical frequency and its Wilson 95\% interval.
\label{tab:sigma-calibration}}
\scriptsize
\begin{tabularx}{\textwidth}{@{}Xlrrrrrrl@{}}
\toprule
Line & End & $|S_0|$ & Ana.\ sd & Emp.\ sd & Ratio & $P_{\mathrm{pred}}$ & $P_{\mathrm{emp}}$ & Wilson 95\% \\
\midrule
"""
        + body
        + "\n"
        + r"""\bottomrule
\end{tabularx}
\end{table}
""",
    )

# ---------------- multi-scale replay (exp3) ----------------
d3 = load("exp3_multiscale_replay.json")
if d3:
    rows = []
    for c in d3["cases"]:
        s = c["crossing_alpha_target_summary"]
        n_viol_at_09 = 0
        n_lines = len(c["per_line"])
        for x in c["per_line"]:
            r09 = next((y for y in x["scales"] if y.get("alpha") == 0.9), None)
            if r09 and r09.get("target_is_violated"):
                n_viol_at_09 += 1
        rows.append(
            f"\\texttt{{{esc(c['case'])}}} & {n_lines} & "
            f"{fixed(s['median'], 3)} & {fixed(s['min'], 3)} & {fixed(s['max'], 3)} & "
            f"{n_viol_at_09}/{n_lines} & {c['q_limit_events_at_base']} \\\\"
        )
    body = "\n".join(rows)
    write(
        "table_replay_multiscale.tex",
        r"""\begin{table}[t]
\centering
\caption{Multi-scale nonlinear replay of the balanced worst-case direction
for the five tightest lines per case, at scales
$\alpha\in\{0.25,0.5,0.75,0.9,1.0,1.1,1.25,1.5\}$ of the affine radius.
``Crossing $\alpha$'' is the interpolated scale at which the TARGET line
first violates its limit in the nonlinear replay; values below 1 quantify
the mild nonconservatism of the affine radius along its own worst
direction.  All replays converged; no voltage-band violations occurred at
$\alpha \le 1$.\label{tab:replay}}
\footnotesize
\begin{tabularx}{\textwidth}{@{}Xrrrrrr@{}}
\toprule
Case & Lines & Med.\ crossing $\alpha$ & Min & Max & Violated at $0.9r$ & Q-lim.\ events (base) \\
\midrule
"""
        + body
        + "\n"
        + r"""\bottomrule
\end{tabularx}
\end{table}
""",
    )

# ---------------- ranking stats (exp4) ----------------
d4 = load("exp4_ranking_stats.json")
if d4:
    nm = {
        "inv_radius": "AC L2 danger score ($1/r$)",
        "loading_ratio": "Loading ratio",
        "inv_headroom": "Headroom danger score",
    }
    rows = []
    for key, label in nm.items():
        sp = d4["spearman"][key]
        p5 = d4["precision_recall_at_k"]["5"][key]
        p10 = d4["precision_recall_at_k"]["10"][key]
        rows.append(
            f"{label} & {fixed(sp['rho'], 3)} & "
            f"[{fixed(sp['ci95_scenario_bootstrap'][0], 3)}, {fixed(sp['ci95_scenario_bootstrap'][1], 3)}] & "
            f"{fixed(p5['precision'], 2)} & {fixed(p10['precision'], 2)} & "
            f"{fixed(p5['top_k_mean_freq'], 3)} \\\\"
        )
    body = "\n".join(rows)
    diffs = d4["paired_differences"]
    dl = diffs["inv_radius_minus_loading_ratio"]
    dh = diffs["inv_radius_minus_inv_headroom"]
    write(
        "table_ranking.tex",
        r"""\begin{table}[t]
\centering
\caption{Ranking statistics on the IEEE 118-bus system with scenario-level
uncertainty quantification ("""
        + f"{d4['n_mc_converged']}/{d4['n_mc_attempted']}"
        + r""" converged Monte Carlo scenarios,
$\sigma_P=\sigma_Q=30$, seed 42).  Confidence intervals are SCENARIO
bootstrap (resampling Monte Carlo scenarios, """
        + f"{d4['n_boot_scenarios']}"
        + r""" replicates), which
treats the correlated line outcomes correctly, unlike the line bootstrap of
the original submission; replicates re-estimate the empirical frequencies, so
the intervals are mildly attenuated relative to the full-sample point
estimate.  The paired scenario-bootstrap 95\% interval for the
Spearman difference is """
        + f"$\\Delta\\rho = {dl['point']:.3f}$, CI $[{dl['ci95'][0]:.3f}, {dl['ci95'][1]:.3f}]$ vs.\\ loading and "
        + f"$\\Delta\\rho = {dh['point']:.3f}$, CI $[{dh['ci95'][0]:.3f}, {dh['ci95'][1]:.3f}]$ vs.\\ headroom"
        + r""".
\label{tab:ranking}}
\footnotesize
\begin{tabularx}{\textwidth}{@{}Xrrrrr@{}}
\toprule
Score & $\rho$ & 95\% CI (scenario) & Prec.@5 & Prec.@10 & Top-5 mean freq. \\
\midrule
"""
        + body
        + "\n"
        + r"""\bottomrule
\end{tabularx}
\end{table}
""",
    )

# ---------------- DC/AC paired (exp7) ----------------
d7 = load("exp7_dc_ac_paired.json")
if d7:
    rows = []
    for c in d7["cases"]:
        if "failed" in c:
            continue
        rows.append(
            f"\\texttt{{{esc(c['case'])}}} & "
            f"{fixed(c['r_dc_global'], 2)} & {fixed(c['r_ac_global'], 2)} & "
            f"{fixed(c['ac_over_dc'], 3)} & {c['n_nondiff_ends']} \\\\"
        )
    body = "\n".join(rows)
    write(
        "table_dc_ac_radii.tex",
        r"""\begin{table}[t]
\centering
\caption{Regenerated paired DC and AC all-constraint radii (ext-grid-consistent
slack, Q-limit-aware active set, operator-norm zero-flow ends; AC certificate
in the balanced two-block $[P;Q]$ Euclidean norm, AC-FPF base points).
\label{tab:dc-ac}}
\footnotesize
\begin{tabularx}{\textwidth}{@{}Xrrrr@{}}
\toprule
Case & $r_\star^{DC}$ & $r_\star^{AC}$ & AC/DC & ND ends \\
\midrule
"""
        + body
        + "\n"
        + r"""\bottomrule
\end{tabularx}
\end{table}
""",
    )

# ---------------- timing (exp5) ----------------
if d5:
    rows = []
    for c in d5["cases"]:
        if "failed" in c:
            rows.append(
                f"\\texttt{{{esc(c['case'])}}} & \\multicolumn{{7}}{{l}}{{lossless runpp base point does not converge (AC-FPF base required, cf.\\ Table~\\ref{{tab:zero-flow}})}} \\\\"
            )
            continue
        rows.append(
            f"\\texttt{{{esc(c['case'])}}} & {c['n_bus']} & {c['n_line']} & "
            f"{c['ac_pf']['mean_s']:.3f}$\\pm${c['ac_pf']['std_s']:.3f} & "
            f"{c['operator_build_assembly_plus_lu']['mean_s']:.3f}$\\pm${c['operator_build_assembly_plus_lu']['std_s']:.3f} & "
            f"{c['certificate_adjoint_plus_norms']['mean_s']:.3f}$\\pm${c['certificate_adjoint_plus_norms']['std_s']:.3f} & "
            f"{c['peak_rss_after_certificate_mb']:.0f} & {c['stored_h_arrays_mb']:.2f} \\\\"
        )
    body = "\n".join(rows)
    write(
        "table_timing.tex",
        r"""\begin{table}[t]
\centering
\caption{Separated, repeated timings (mean $\pm$ std over 7 repetitions) and
measured peak resident memory.  Stages: nonlinear AC PF; operator build
(Jacobian assembly + sparse LU); certificate (chunked adjoint solves +
balanced norms, including the adjoint-residual check).
\label{tab:timing}}
\scriptsize
\begin{tabularx}{\textwidth}{@{}Xrrrrrrr@{}}
\toprule
Case & Buses & Lines & AC PF (s) & Operator (s) & Certificate (s) & Peak RSS (MB) & Stored $h$ (MB) \\
\midrule
"""
        + body
        + "\n"
        + r"""\bottomrule
\end{tabularx}
\end{table}
""",
    )

# ---------------- zero flow (exp6) ----------------
d6 = load("exp6_zero_flow_case2000.json")
if d6:
    sens = d6["nd_threshold_sensitivity_end_counts"]
    sens_str = ", ".join(f"$\\le {k}$: {v}" for k, v in sens.items())
    write(
        "table_zero_flow.tex",
        r"""\begin{table}[t]
\centering
\caption{Zero-flow line ends on \texttt{case2000\_goc} (AC-FPF base point).
Every nondifferentiable end now receives a first-order operator-norm radius,
so the all-constraint certificate is defined; previously these ends were
excluded and the case was only partially certified.  End counts with
$|S_0|$ below a threshold $t$ (MVA): """
        + sens_str
        + r""" --- the count is insensitive to the
threshold over eight orders of magnitude, supporting the scale-aware default.
\label{tab:zero-flow}}
\footnotesize
\begin{tabularx}{\textwidth}{@{}Xr@{}}
\toprule
Quantity & Value \\
\midrule
Monitored lines & """
        + str(d6["n_lines"])
        + r""" \\
Lines with a zero-flow end & """
        + str(d6["n_nd_lines"])
        + r""" \\
Operator-norm certified & """
        + str(d6["n_operator_norm_certified"])
        + r""" \\
All-constraint radius defined & """
        + ("yes" if d6["all_constraint_radius_now_defined"] else "no")
        + r""" \\
Global minimum radius & """
        + f"{d6['global_min_radius']:.2f}"
        + r""" \\
Q-limit-saturated buses absorbed as PQ & """
        + str(d6["q_limit_events_at_base"])
        + r""" \\
Max adjoint residual & """
        + sci(d6["adjoint_residual_max"])
        + r""" \\
\bottomrule
\end{tabularx}
\end{table}
""",
    )


if __name__ == "__main__":
    pass
