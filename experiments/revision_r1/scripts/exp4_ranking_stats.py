"""Experiment R1-4 (reviewer point 10): ranking statistics done properly.

case118, isotropic balanced Gaussian P/Q errors (sigma = 30 MW / 30 MVAr, the
paper's setting), N Monte Carlo scenarios.  Danger scores: 1/r (AC L2),
loading ratio, 1/headroom.  Statistics:
  - Spearman rho per score with SCENARIO-bootstrap 95% CIs
    (resampling scenarios, not lines);
  - paired bootstrap CIs for the DIFFERENCE in rho (radius vs baselines);
  - precision@k / recall@k for identifying the top-risk lines;
  - false negatives at k: overloaded lines missed by each score's top-k.
"""

from __future__ import annotations

import copy
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PF_KW, certificate_with_h, load_case, save_json  # noqa: E402

import pandapower as pp  # noqa: E402
from scipy import stats  # noqa: E402

from stability_radius.verification.sampling import (  # noqa: E402
    sample_balanced_gaussian_sigma,
)

CASE = "pglib_opf_case118_ieee.m"
SEED = 42
N_MC = 8000
N_BOOT = 2000
_SAMPLE_TIMEOUT_S = 10  # skip pathological Q-limit flip-flop samples
SIGMA_P = 30.0
SIGMA_Q = 30.0
K_LIST = [3, 5, 10]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(stats.spearmanr(x, y).statistic)


def main() -> None:
    t0 = time.time()
    net, slack = load_case(CASE)
    bus_ids = [int(x) for x in sorted(net.bus.index)]
    line_ids = [int(x) for x in sorted(net.line.index)]
    m = len(line_ids)

    base_pf, ac, hv, h_from, h_to, _ = certificate_with_h(net, slack)
    constrained = np.array(
        [not ac[f"line_{lid}"]["is_unconstrained"] for lid in line_ids]
    )
    limits = np.array([float(ac[f"line_{lid}"]["ac_s_limit_mva"]) for lid in line_ids])
    s0 = np.array(
        [
            max(
                float(ac[f"line_{lid}"]["ac_s0_from_mva"]),
                float(ac[f"line_{lid}"]["ac_s0_to_mva"]),
            )
            for lid in line_ids
        ]
    )
    radius = np.array([float(ac[f"line_{lid}"]["radius_ac_l2"]) for lid in line_ids])

    scores = {
        "inv_radius": np.where(radius > 0, 1.0 / radius, np.inf),
        "loading_ratio": s0 / limits,
        "inv_headroom": 1.0 / np.maximum(limits - s0, 1e-9),
    }

    # ---------------- Monte Carlo ----------------
    rng = np.random.default_rng(SEED)
    sig_p = np.full(len(bus_ids), SIGMA_P)
    sig_q = np.full(len(bus_ids), SIGMA_Q)
    dP, dQ = sample_balanced_gaussian_sigma(
        rng=rng, n=N_MC, sigma_p=sig_p, sigma_q=sig_q
    )

    base = copy.deepcopy(net)
    pp.runpp(base, init="dc", **PF_KW)
    sg = [pp.create_sgen(base, b, p_mw=0.0, q_mvar=0.0) for b in bus_ids]

    def _alarm(_sig, _frm):
        raise TimeoutError

    signal.signal(signal.SIGALRM, _alarm)

    over = np.zeros((N_MC, m), dtype=bool)
    ok = np.zeros(N_MC, dtype=bool)
    n_timeout = 0
    for k in range(N_MC):
        base.sgen.loc[sg, "p_mw"] = dP[k]
        base.sgen.loc[sg, "q_mvar"] = dQ[k]
        signal.alarm(_SAMPLE_TIMEOUT_S)
        try:
            # independent init per sample: chaining init="results" can push the
            # Q-limit outer loop into a persistent flip-flop on rare samples
            pp.runpp(base, init="dc", **PF_KW)
        except TimeoutError:
            n_timeout += 1
            continue
        except Exception:
            continue
        finally:
            signal.alarm(0)
        ok[k] = True
        s_all = np.maximum(
            np.hypot(base.res_line.p_from_mw.values, base.res_line.q_from_mvar.values),
            np.hypot(base.res_line.p_to_mw.values, base.res_line.q_to_mvar.values),
        )
        over[k] = (s_all > limits) & constrained
        if (k + 1) % 1000 == 0:
            print(f"  {k + 1}/{N_MC} samples, {time.time() - t0:.0f}s", flush=True)

    over = over[ok]
    n_eff = int(over.shape[0])
    freq = over.mean(axis=0)

    # ---------------- Spearman + scenario bootstrap ----------------
    rho = {name: spearman(sc, freq) for name, sc in scores.items()}

    boot = {name: np.empty(N_BOOT) for name in scores}
    rng_b = np.random.default_rng(SEED + 1)
    for b in range(N_BOOT):
        idx = rng_b.integers(0, n_eff, n_eff)
        fb = over[idx].mean(axis=0)
        for name, sc in scores.items():
            boot[name][b] = spearman(sc, fb)

    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    diffs = {
        f"inv_radius_minus_{other}": {
            "point": float(rho["inv_radius"] - rho[other]),
            "ci95": ci(boot["inv_radius"] - boot[other]),
        }
        for other in ("loading_ratio", "inv_headroom")
    }

    # ---------------- precision/recall @ k, FN ----------------
    risk_rank = np.argsort(-freq)
    pr = {}
    for kk in K_LIST:
        actual_top = set(np.array(line_ids)[risk_rank[:kk]].tolist())
        pr[str(kk)] = {}
        for name, sc in scores.items():
            pred_top = set(np.array(line_ids)[np.argsort(-sc)[:kk]].tolist())
            tp = len(actual_top & pred_top)
            pr[str(kk)][name] = {
                "precision": tp / kk,
                "recall": tp / kk,
                "false_negative_lines": sorted(actual_top - pred_top),
                "top_k_mean_freq": float(freq[np.argsort(-sc)[:kk]].mean()),
            }

    out = {
        "experiment": "ranking_stats_case118",
        "seed": SEED,
        "n_mc_attempted": N_MC,
        "n_mc_converged": n_eff,
        "n_mc_sample_timeouts": n_timeout,
        "sigma_p_mw": SIGMA_P,
        "sigma_q_mvar": SIGMA_Q,
        "n_boot_scenarios": N_BOOT,
        "spearman": {
            name: {"rho": rho[name], "ci95_scenario_bootstrap": ci(boot[name])}
            for name in scores
        },
        "paired_differences": diffs,
        "precision_recall_at_k": pr,
        "n_lines_with_nonzero_freq": int((freq > 0).sum()),
        "runtime_s": float(time.time() - t0),
    }
    for name in scores:
        print(
            f"{name:14s} rho={rho[name]:.3f} CI={out['spearman'][name]['ci95_scenario_bootstrap']}"
        )
    for k_, v in diffs.items():
        print(f"{k_}: {v['point']:.3f} CI={v['ci95']}")
    save_json("exp4_ranking_stats.json", out)


if __name__ == "__main__":
    main()
