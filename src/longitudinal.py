#!/usr/bin/env python3
"""
Subject-level longitudinal analysis of reduced Wilson-Cowan dynamics.

Primary input
-------------
outputs/cohort_dynamics_40_primary/cohort_dynamics_800.csv

The input is expected to contain the inference-valid chunk set produced after
applying the frozen regression-assisted selector. Chunks remain repeated
observations; all formal longitudinal tests are performed on one summary per
mouse per session.

Analyses
--------
1. Recompute the equilibrium structure of the selected reduced model
       tau dE/dt = -E + sigmoid(w_EE E + h)
   and identify the STABLE equilibrium nearest the observed chunk mean.
2. Aggregate chunk-level results to 10 mice x 4 sessions.
3. Quantify per mouse/session:
   - fraction of chunks with model-implied bistability,
   - median w_EE and h,
   - median distance to the nearest saddle-node fold where defined,
   - median distance to the cusp (w=4, h=-2),
   - median local eigenvalue of the nearest STABLE equilibrium,
   - fractions of fold-defined chunks within several fold-distance thresholds.
4. Test session effects using Friedman repeated-measures tests on mouse-level
   summaries. Pairwise Wilcoxon signed-rank tests are Holm-corrected within each
   outcome family.

Guardrail
---------
"Bistable" and related labels describe the inferred reduced model in
normalized-fluorescence coordinates. They are model-implied dynamical regimes,
not direct measurements of physiological cortical states.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SESSION_ORDER = ["day0", "day1", "day7", "day15"]
EXPECTED_SUBJECTS = 10
EXPECTED_SESSIONS = 4
TAU_DEFAULT = 1.0
FOLD_THRESHOLDS = [0.01, 0.025, 0.05, 0.10]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/cohort_dynamics_40_primary/cohort_dynamics_800.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cohort_longitudinal_40_primary"),
    )
    p.add_argument("--tau", type=float, default=TAU_DEFAULT)
    return p.parse_args()


def logit(e):
    e = np.asarray(e, dtype=float)
    return np.log(e / (1.0 - e))


def equilibrium_function(e, w, h):
    return float(logit(e) - w * e - h)


def fold_points(w):
    if w <= 4.0:
        return None
    disc = np.sqrt(1.0 - 4.0 / w)
    e1 = 0.5 * (1.0 - disc)
    e2 = 0.5 * (1.0 + disc)
    pts = []
    for e in (e1, e2):
        h = float(logit(e) - w * e)
        pts.append((float(e), h))
    pts.sort(key=lambda x: x[1])
    return pts


def find_equilibria(w, h, tol=1e-11):
    eps = 1e-10
    cuts = [eps]
    folds = fold_points(w)
    if folds is not None:
        cuts.extend(sorted([folds[0][0], folds[1][0]]))
    cuts.append(1.0 - eps)
    cuts = sorted(set(cuts))

    roots = []
    for x in cuts[1:-1]:
        fx = equilibrium_function(x, w, h)
        if abs(fx) < 1e-8:
            roots.append(float(x))

    for a, b in zip(cuts[:-1], cuts[1:]):
        aa = a + 1e-9 if a > eps else a
        bb = b - 1e-9 if b < 1.0 - eps else b
        if aa >= bb:
            continue
        fa = equilibrium_function(aa, w, h)
        fb = equilibrium_function(bb, w, h)
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0.0:
            roots.append(
                float(brentq(equilibrium_function, aa, bb, args=(w, h), xtol=tol))
            )

    roots = sorted(roots)
    unique = []
    for r in roots:
        if not unique or abs(r - unique[-1]) > 1e-7:
            unique.append(r)
    return unique


def local_lambda(e, w, tau):
    return float((-1.0 + w * e * (1.0 - e)) / tau)


def stable_equilibrium_metrics(w, h, state_mean, tau):
    roots = find_equilibria(w, h)
    if not roots:
        raise RuntimeError(f"No equilibrium found for w={w}, h={h}")

    vals = []
    for e in roots:
        lam = local_lambda(e, w, tau)
        label = "stable" if lam < -1e-8 else ("unstable" if lam > 1e-8 else "neutral")
        vals.append((e, lam, label))

    stable = [v for v in vals if v[2] == "stable"]
    if not stable:
        # This should only occur at an exact/near fold under this scalar model.
        nonpositive = [v for v in vals if v[1] <= 1e-8]
        stable = nonpositive if nonpositive else vals

    nearest = min(stable, key=lambda z: abs(z[0] - state_mean))
    n_stable = sum(v[2] == "stable" for v in vals)
    regime = "bistable" if len(vals) == 3 and n_stable == 2 else "monostable"
    if any(v[2] == "neutral" for v in vals):
        regime = "near_fold"

    return {
        "recomputed_n_equilibria": len(vals),
        "recomputed_n_stable_equilibria": n_stable,
        "recomputed_regime": regime,
        "nearest_stable_equilibrium": float(nearest[0]),
        "nearest_stable_lambda0": float(nearest[1]),
        "state_mean_minus_nearest_stable_equilibrium": float(state_mean - nearest[0]),
        "nearest_stable_abs_state_distance": float(abs(state_mean - nearest[0])),
    }


def add_stable_metrics(df, tau):
    rows = []
    for _, r in df.iterrows():
        m = stable_equilibrium_metrics(
            float(r["selected_w_EE"]),
            float(r["selected_h"]),
            float(r["state_mean_scaled"]),
            tau,
        )
        row = r.to_dict()
        row.update(m)
        rows.append(row)
    out = pd.DataFrame(rows)

    # The recomputed scalar-model regime should agree with the Phase 3 label.
    if "dynamical_regime" in out.columns:
        comparable = out["dynamical_regime"].isin(["monostable", "bistable", "near_fold"])
        mismatch = comparable & (out["dynamical_regime"].astype(str) != out["recomputed_regime"])
        if mismatch.any():
            bad = out.loc[mismatch, ["subject", "session", "chunk_idx", "dynamical_regime", "recomputed_regime"]]
            raise RuntimeError(f"Regime recomputation mismatch in {len(bad)} rows:\n{bad.head(20)}")
    return out


def safe_fraction(mask):
    mask = pd.Series(mask).dropna()
    return float(mask.mean()) if len(mask) else np.nan


def subject_session_summary(df):
    rows = []
    for (subject, session), g in df.groupby(["subject", "session"], observed=True, sort=True):
        fold = g["distance_to_nearest_fold_h"].dropna()
        row = {
            "subject": subject,
            "session": session,
            "n_valid_chunks": int(len(g)),
            "n_fold_defined_chunks": int(len(fold)),
            "fraction_bistable": float((g["recomputed_regime"] == "bistable").mean()),
            "fraction_monostable": float((g["recomputed_regime"] == "monostable").mean()),
            "median_selected_w_EE": float(g["selected_w_EE"].median()),
            "mean_selected_w_EE": float(g["selected_w_EE"].mean()),
            "median_selected_h": float(g["selected_h"].median()),
            "mean_selected_h": float(g["selected_h"].mean()),
            "median_distance_to_cusp_wh": float(g["distance_to_cusp_wh"].median()),
            "mean_distance_to_cusp_wh": float(g["distance_to_cusp_wh"].mean()),
            "median_nearest_stable_lambda0": float(g["nearest_stable_lambda0"].median()),
            "mean_nearest_stable_lambda0": float(g["nearest_stable_lambda0"].mean()),
            "median_abs_state_distance_to_nearest_stable_equilibrium": float(
                g["nearest_stable_abs_state_distance"].median()
            ),
            "median_distance_to_nearest_fold_h": float(fold.median()) if len(fold) else np.nan,
            "mean_distance_to_nearest_fold_h": float(fold.mean()) if len(fold) else np.nan,
        }
        for t in FOLD_THRESHOLDS:
            key = f"fraction_fold_defined_within_{str(t).replace('.', 'p')}"
            row[key] = float((fold <= t).mean()) if len(fold) else np.nan
        rows.append(row)

    ss = pd.DataFrame(rows)
    ss["session"] = pd.Categorical(ss["session"], SESSION_ORDER, ordered=True)
    return ss.sort_values(["subject", "session"]).reset_index(drop=True)


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj_sorted[rank] = min(1.0, running)
    adj = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        adj[idx] = adj_sorted[rank]
    return adj


def rank_biserial_from_differences(diff):
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = rankdata(np.abs(d), method="average")
    w_pos = float(ranks[d > 0].sum())
    w_neg = float(ranks[d < 0].sum())
    denom = w_pos + w_neg
    return (w_pos - w_neg) / denom if denom else 0.0


def metric_wide(ss, metric):
    w = ss.pivot(index="subject", columns="session", values=metric)
    return w.reindex(columns=SESSION_ORDER)


def friedman_for_metric(ss, metric):
    w = metric_wide(ss, metric).dropna()
    result = {
        "metric": metric,
        "n_complete_mice": int(len(w)),
        "friedman_chi2": np.nan,
        "friedman_p": np.nan,
    }
    if len(w) >= 3:
        arrays = [w[s].to_numpy(float) for s in SESSION_ORDER]
        try:
            stat, p = friedmanchisquare(*arrays)
            result["friedman_chi2"] = float(stat)
            result["friedman_p"] = float(p)
        except ValueError:
            # e.g. all values identical across all sessions
            result["friedman_chi2"] = 0.0
            result["friedman_p"] = 1.0
    return result


def pairwise_for_metric(ss, metric):
    wide = metric_wide(ss, metric)
    rows = []
    for a, b in itertools.combinations(SESSION_ORDER, 2):
        pair = wide[[a, b]].dropna()
        n = len(pair)
        if n == 0:
            rows.append({
                "metric": metric, "session_a": a, "session_b": b,
                "n_paired_mice": 0, "median_a": np.nan, "median_b": np.nan,
                "median_paired_change_b_minus_a": np.nan,
                "wilcoxon_statistic": np.nan, "p_raw": np.nan,
                "rank_biserial_b_minus_a": np.nan,
            })
            continue
        x = pair[a].to_numpy(float)
        y = pair[b].to_numpy(float)
        d = y - x
        if np.allclose(d, 0.0):
            stat, p = 0.0, 1.0
        else:
            try:
                res = wilcoxon(y, x, alternative="two-sided", zero_method="wilcox", method="auto")
                stat, p = float(res.statistic), float(res.pvalue)
            except ValueError:
                stat, p = 0.0, 1.0
        rows.append({
            "metric": metric,
            "session_a": a,
            "session_b": b,
            "n_paired_mice": int(n),
            "median_a": float(np.median(x)),
            "median_b": float(np.median(y)),
            "median_paired_change_b_minus_a": float(np.median(d)),
            "wilcoxon_statistic": stat,
            "p_raw": p,
            "rank_biserial_b_minus_a": float(rank_biserial_from_differences(d)),
        })
    out = pd.DataFrame(rows)
    finite = out["p_raw"].notna()
    out["p_holm"] = np.nan
    if finite.any():
        out.loc[finite, "p_holm"] = holm_adjust(out.loc[finite, "p_raw"].to_numpy(float))
    return out


def session_descriptive(ss):
    metrics = [
        "fraction_bistable",
        "median_selected_w_EE",
        "median_selected_h",
        "median_distance_to_nearest_fold_h",
        "median_distance_to_cusp_wh",
        "median_nearest_stable_lambda0",
    ]
    rows = []
    for session in SESSION_ORDER:
        g = ss[ss["session"].astype(str) == session]
        for metric in metrics:
            x = g[metric].dropna().to_numpy(float)
            rows.append({
                "session": session,
                "metric": metric,
                "n_mice": int(len(x)),
                "mean": float(np.mean(x)) if len(x) else np.nan,
                "sd": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
                "median": float(np.median(x)) if len(x) else np.nan,
                "q25": float(np.quantile(x, 0.25)) if len(x) else np.nan,
                "q75": float(np.quantile(x, 0.75)) if len(x) else np.nan,
                "min": float(np.min(x)) if len(x) else np.nan,
                "max": float(np.max(x)) if len(x) else np.nan,
            })
    return pd.DataFrame(rows)


def make_spaghetti(ss, metric, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    x = np.arange(len(SESSION_ORDER))
    for subject, g in ss.groupby("subject", observed=True):
        vals = (
            g.set_index(g["session"].astype(str))[metric]
            .reindex(SESSION_ORDER)
            .to_numpy(float)
        )
        ax.plot(x, vals, marker="o", linewidth=1.0, alpha=0.55, label=str(subject))
    med = ss.assign(session_s=ss["session"].astype(str)).groupby("session_s", observed=True)[metric].median().reindex(SESSION_ORDER)
    ax.plot(x, med.to_numpy(float), marker="o", linewidth=3.0, label="cohort median")
    ax.set_xticks(x)
    ax.set_xticklabels(SESSION_ORDER)
    ax.set_xlabel("Session")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def make_parameter_plane(ss, path):
    fig, ax = plt.subplots(figsize=(7.3, 5.6))
    for subject, g in ss.groupby("subject", observed=True):
        g = g.copy()
        g["session_s"] = g["session"].astype(str)
        g = g.set_index("session_s").reindex(SESSION_ORDER)
        ax.plot(g["median_selected_w_EE"], g["median_selected_h"], marker="o", linewidth=1.0, alpha=0.65, label=str(subject))
    ax.scatter([4.0], [-2.0], marker="*", s=100, label="cusp (4, -2)")
    ax.set_xlabel(r"Median $w_{EE}$")
    ax.set_ylabel(r"Median $h$")
    ax.set_title("Mouse-level longitudinal trajectories in parameter space")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main():
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    required = {
        "subject", "session", "chunk_idx", "selected_w_EE", "selected_h",
        "state_mean_scaled", "distance_to_nearest_fold_h", "distance_to_cusp_wh",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input missing required columns: {sorted(missing)}")

    subjects = sorted(df["subject"].astype(str).unique())
    sessions = sorted(df["session"].astype(str).unique())
    if len(subjects) != EXPECTED_SUBJECTS:
        raise ValueError(f"Expected {EXPECTED_SUBJECTS} subjects, found {len(subjects)}")
    if set(sessions) != set(SESSION_ORDER):
        raise ValueError(f"Expected sessions {SESSION_ORDER}, found {sessions}")

    # Corrected chunk-level stability metric: nearest STABLE equilibrium.
    chunk = add_stable_metrics(df, args.tau)
    chunk.to_csv(out / "chunk_dynamics_nearest_stable_equilibrium.csv", index=False)

    ss = subject_session_summary(chunk)
    if len(ss) != EXPECTED_SUBJECTS * EXPECTED_SESSIONS:
        raise ValueError(f"Expected 40 subject-session summaries, found {len(ss)}")
    ss.to_csv(out / "subject_session_dynamics_40.csv", index=False)

    descriptive = session_descriptive(ss)
    descriptive.to_csv(out / "session_mouse_level_descriptive.csv", index=False)

    metrics = [
        "fraction_bistable",
        "median_selected_w_EE",
        "median_selected_h",
        "median_distance_to_nearest_fold_h",
        "median_distance_to_cusp_wh",
        "median_nearest_stable_lambda0",
    ]

    friedman_rows = [friedman_for_metric(ss, m) for m in metrics]
    friedman = pd.DataFrame(friedman_rows)
    friedman.to_csv(out / "friedman_session_effects.csv", index=False)

    pairwise = pd.concat([pairwise_for_metric(ss, m) for m in metrics], ignore_index=True)
    pairwise.to_csv(out / "wilcoxon_pairwise_holm.csv", index=False)

    # Wide mouse-level table makes individual longitudinal trajectories auditable.
    wide_parts = []
    for metric in metrics:
        w = metric_wide(ss, metric)
        w.columns = [f"{metric}_{c}" for c in w.columns]
        wide_parts.append(w)
    subject_wide = pd.concat(wide_parts, axis=1).reset_index()
    subject_wide.to_csv(out / "subject_longitudinal_wide_10.csv", index=False)

    make_spaghetti(
        ss, "fraction_bistable", "Fraction bistable",
        "Mouse-level longitudinal model-implied bistability",
        out / "fraction_bistable_by_mouse_session.png",
    )
    make_spaghetti(
        ss, "median_distance_to_nearest_fold_h", "Median distance to nearest fold in h",
        "Mouse-level longitudinal proximity to saddle-node boundary",
        out / "fold_distance_by_mouse_session.png",
    )
    make_spaghetti(
        ss, "median_nearest_stable_lambda0", r"Median nearest-stable $\lambda(0)$",
        "Local stability of nearest stable equilibrium",
        out / "nearest_stable_lambda_by_mouse_session.png",
    )
    make_parameter_plane(ss, out / "mouse_session_parameter_trajectories.png")

    # Compact machine-readable summary.
    f_bist = friedman.loc[friedman["metric"] == "fraction_bistable"].iloc[0].to_dict()
    summary = {
        "input_rows": int(len(df)),
        "n_subjects": int(len(subjects)),
        "n_subject_sessions": int(len(ss)),
        "formal_replication_unit": "mouse",
        "session_order": SESSION_ORDER,
        "primary_outcome": "fraction_bistable per mouse-session",
        "primary_friedman": {
            "n_complete_mice": int(f_bist["n_complete_mice"]),
            "chi2": None if pd.isna(f_bist["friedman_chi2"]) else float(f_bist["friedman_chi2"]),
            "p": None if pd.isna(f_bist["friedman_p"]) else float(f_bist["friedman_p"]),
        },
        "guardrails": [
            "Formal tests are performed on mouse-level session summaries, not on 738 chunk rows as independent replicates.",
            "Bistability is model-implied in normalized-fluorescence coordinates and is not a direct physiological state label.",
            "Pairwise Wilcoxon p-values are Holm-corrected within each outcome family.",
            "Nearest-equilibrium stability was recomputed using the nearest STABLE equilibrium, correcting the earlier descriptive metric that could select the unstable middle branch in bistable chunks.",
        ],
    }
    (out / "longitudinal_analysis_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print("\nFriedman tests:")
    print(friedman.to_string(index=False))
    print(f"\nOutputs saved to {out}")


if __name__ == "__main__":
    main()
