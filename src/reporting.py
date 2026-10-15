#!/usr/bin/env python3
"""
Build manuscript-ready figures, tables, and an auto-filled Results draft for the
40-recording longitudinal Wilson-Cowan cohort extension.

This script does NOT refit the inverse model. It synthesizes already-completed
Phase 2/3/3B/3C outputs while preserving the inferential hierarchy:
  * mouse-level repeated-measures tests are primary for longitudinal inference;
  * GEE/mixed-model session tests are supporting;
  * chunks are repeated observations, not independent biological replicates.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SESSION_ORDER = ["day0", "day1", "day7", "day15"]
FOLD_THRESHOLDS = [0.01, 0.025, 0.05, 0.10]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", type=Path,
                   default=Path("outputs/cohort_model_selection_40/cohort_model_selection_800.csv"))
    p.add_argument("--dynamics", type=Path,
                   default=Path("outputs/cohort_dynamics_40_primary/cohort_dynamics_800.csv"))
    p.add_argument("--longitudinal-chunks", type=Path,
                   default=Path("outputs/cohort_longitudinal_40_primary/chunk_dynamics_nearest_stable_equilibrium.csv"))
    p.add_argument("--subject-session", type=Path,
                   default=Path("outputs/cohort_longitudinal_40_primary/subject_session_dynamics_40.csv"))
    p.add_argument("--friedman", type=Path,
                   default=Path("outputs/cohort_longitudinal_40_primary/friedman_session_effects.csv"))
    p.add_argument("--wilcoxon", type=Path,
                   default=Path("outputs/cohort_longitudinal_40_primary/wilcoxon_pairwise_holm.csv"))
    p.add_argument("--variance", type=Path,
                   default=Path("outputs/cohort_hierarchical_40_primary/continuous_variance_components.csv"))
    p.add_argument("--gee", type=Path,
                   default=Path("outputs/cohort_hierarchical_40_primary/bistability_GEE_session_effects.csv"))
    p.add_argument("--hier-summary", type=Path,
                   default=Path("outputs/cohort_hierarchical_40_primary/hierarchical_variance_summary.json"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/manuscript_synthesis_40"))
    return p.parse_args()


def require(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    out = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m-rank)*p[idx])
        out[idx] = min(1.0, running)
    return out


def rank_biserial(diff):
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = rankdata(np.abs(d), method="average")
    pos = float(ranks[d > 0].sum())
    neg = float(ranks[d < 0].sum())
    return (pos-neg)/(pos+neg) if pos+neg else 0.0


def state_distance_mouse_tests(long_df: pd.DataFrame):
    metric = "nearest_stable_abs_state_distance"
    ss = (long_df.groupby(["subject", "session"], observed=True)[metric]
          .median().unstack("session").reindex(columns=SESSION_ORDER))
    complete = ss.dropna()
    stat, p = friedmanchisquare(*[complete[s].to_numpy(float) for s in SESSION_ORDER])
    rows = []
    for a, b in combinations(SESSION_ORDER, 2):
        pair = complete[[a,b]].dropna()
        x = pair[a].to_numpy(float)
        y = pair[b].to_numpy(float)
        d = y-x
        if np.allclose(d, 0):
            w, pr = 0.0, 1.0
        else:
            res = wilcoxon(y, x, alternative="two-sided", zero_method="wilcox", method="auto")
            w, pr = float(res.statistic), float(res.pvalue)
        rows.append({
            "metric": metric,
            "session_a": a, "session_b": b,
            "n_paired_mice": int(len(pair)),
            "median_a": float(np.median(x)),
            "median_b": float(np.median(y)),
            "median_paired_change_b_minus_a": float(np.median(d)),
            "wilcoxon_statistic": w,
            "p_raw": pr,
            "rank_biserial_b_minus_a": float(rank_biserial(d)),
        })
    pair = pd.DataFrame(rows)
    pair["p_holm"] = holm_adjust(pair["p_raw"].to_numpy(float))
    session_med = complete.median(axis=0).reindex(SESSION_ORDER)
    summary = {
        "n_complete_mice": int(len(complete)),
        "friedman_chi2": float(stat),
        "friedman_p": float(p),
        "session_medians": {s: float(session_med[s]) for s in SESSION_ORDER},
    }
    return complete, pair, summary


def fold_points(w):
    if w <= 4.0:
        return None
    disc = np.sqrt(1.0 - 4.0/w)
    es = [0.5*(1-disc), 0.5*(1+disc)]
    pts = []
    for e in es:
        h = np.log(e/(1-e)) - w*e
        pts.append((float(e), float(h)))
    pts.sort(key=lambda z: z[1])
    return pts


def fold_curve(wmax):
    rows=[]
    for w in np.linspace(4.0001, max(4.2, wmax), 1200):
        pts=fold_points(float(w))
        if pts:
            rows.append((w, pts[0][1], "lower"))
            rows.append((w, pts[1][1], "upper"))
    return pd.DataFrame(rows, columns=["w_EE","h","branch"])


def save_both(fig, base: Path):
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def figure_criterion_rates(cohort, out):
    cols = [c for c in cohort.columns if c.startswith("criterion_")]
    labels = [c.replace("criterion_", "").replace("_", " ") for c in cols]
    rates = [100*pd.to_numeric(cohort[c], errors="coerce").fillna(False).astype(bool).mean() for c in cols]
    order = np.argsort(rates)
    fig, ax = plt.subplots(figsize=(8.2,6.0))
    ax.barh(np.array(labels)[order], np.array(rates)[order])
    ax.axvline(100, linewidth=0.7)
    ax.set_xlabel("Chunks passing criterion (%)")
    ax.set_ylabel("")
    ax.set_title("Frozen model-selection criteria across 800 chunks")
    ax.set_xlim(0, 103)
    for i, val in enumerate(np.array(rates)[order]):
        ax.text(min(val+1, 99.5), i, f"{val:.1f}%", va="center", fontsize=8)
    save_both(fig, out/"figure1_model_selection_criterion_pass_rates")


def figure_parameter_plane(long_df, out):
    wmax = float(long_df["selected_w_EE"].max()) + 0.15
    fc = fold_curve(wmax)
    fig, ax = plt.subplots(figsize=(7.4,5.8))
    for branch, g in fc.groupby("branch"):
        ax.plot(g["w_EE"], g["h"], linewidth=1.4, label=f"saddle-node fold ({branch})")
    regimes = ["monostable","bistable"]
    for reg in regimes:
        g = long_df[long_df["recomputed_regime"].astype(str)==reg]
        if len(g):
            ax.scatter(g["selected_w_EE"], g["selected_h"], s=16, alpha=0.5, label=reg)
    ax.scatter([4],[-2], marker="*", s=120, label="cusp")
    ax.set_xlabel(r"Selected $w_{EE}$")
    ax.set_ylabel(r"Selected $h$")
    ax.set_title("Inference-valid reduced models in bifurcation parameter space")
    ax.legend(fontsize=8)
    save_both(fig, out/"figure2_bifurcation_parameter_plane")


def figure_spaghetti(ss, metric, ylabel, title, outbase, annotate=None):
    fig, ax = plt.subplots(figsize=(7.2,5.3))
    x = np.arange(4)
    for subject, g in ss.groupby("subject", observed=True):
        vals=(g.assign(session_s=g["session"].astype(str)).set_index("session_s")[metric]
              .reindex(SESSION_ORDER).to_numpy(float))
        ax.plot(x, vals, marker="o", linewidth=0.9, alpha=0.5)
    med=(ss.assign(session_s=ss["session"].astype(str)).groupby("session_s", observed=True)[metric]
         .median().reindex(SESSION_ORDER))
    ax.plot(x, med.to_numpy(float), marker="o", linewidth=3.0, label="cohort median")
    ax.set_xticks(x); ax.set_xticklabels(SESSION_ORDER)
    ax.set_xlabel("Session"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8)
    if annotate:
        ax.text(0.02, 0.98, annotate, transform=ax.transAxes, va="top", fontsize=8)
    save_both(fig, outbase)


def figure_variance_partition(vc, out):
    keep = vc.copy()
    labelmap = {
        "selected_w_EE": r"$w_{EE}$",
        "selected_h": r"$h$",
        "distance_to_cusp_wh": "cusp distance",
        "nearest_stable_lambda0": r"nearest-stable $\lambda_0$",
        "nearest_stable_abs_state_distance": "state-to-stable distance",
        "distance_to_nearest_fold_h": "fold distance",
    }
    keep["display"] = keep["outcome"].map(labelmap).fillna(keep["outcome"])
    y=np.arange(len(keep))
    a=100*keep["mouse_variance_fraction"].to_numpy(float)
    b=100*keep["mouse_session_variance_fraction"].to_numpy(float)
    c=100*keep["residual_chunk_variance_fraction"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8.1,5.6))
    ax.barh(y, a, label="mouse")
    ax.barh(y, b, left=a, label="mouse-session")
    ax.barh(y, c, left=a+b, label="chunk/residual")
    ax.set_yticks(y); ax.set_yticklabels(keep["display"])
    ax.set_xlabel("Variance component (%)")
    ax.set_title("Hierarchical variance partition of reduced-model dynamics")
    ax.set_xlim(0,100)
    ax.legend(fontsize=8, loc="lower right")
    save_both(fig, out/"figure5_hierarchical_variance_partition")


def compact_tables(cohort, long_df, ss, friedman, pairwise, vc, gee, state_pair, state_summary, out):
    # Table 1: model selection and evidence.
    valid = cohort[cohort["criterion_valid_fraction"].astype(bool) & cohort["criterion_conditioning"].astype(bool)]
    rows=[]
    for name,g in [("All chunks",cohort),("Inference-valid",valid)]:
        rows.append({
            "analysis_set":name,
            "n_chunks":len(g),
            "n_reduced":int((g["selected_model"]=="reduced").sum()),
            "n_full":int((g["selected_model"]=="full").sum()),
            "positive_unrestricted_D":int((g["full_regression_D_E"]>0).sum()),
            "positive_D_bootstrap_CI":int((g["D_boot_ci95_low"]>0).sum()),
            "logit_gain_ge_0p1pct":int((g["nested_logit_improvement_pct"]>=0.1).sum()),
            "pde_gain_ge_0p1pct":int((g["nested_pde_improvement_pct"]>=0.1).sum()),
        })
    pd.DataFrame(rows).to_csv(out/"table1_model_selection_summary.csv",index=False)

    # Table 2: session dynamics.
    dyn_rows=[]
    for s in SESSION_ORDER:
        g=long_df[long_df["session"].astype(str)==s]
        fold=g["distance_to_nearest_fold_h"].dropna()
        dyn_rows.append({
            "session":s,"n_valid_chunks":len(g),
            "n_bistable":int((g["recomputed_regime"]=="bistable").sum()),
            "fraction_bistable":float((g["recomputed_regime"]=="bistable").mean()),
            "median_selected_w_EE":float(g["selected_w_EE"].median()),
            "median_selected_h":float(g["selected_h"].median()),
            "median_distance_to_cusp_wh":float(g["distance_to_cusp_wh"].median()),
            "median_distance_to_nearest_fold_h_when_defined":float(fold.median()) if len(fold) else np.nan,
            "median_state_to_nearest_stable_equilibrium":float(g["nearest_stable_abs_state_distance"].median()),
        })
    pd.DataFrame(dyn_rows).to_csv(out/"table2_session_dynamics_summary.csv",index=False)

    # Table 3: primary longitudinal tests + exploratory state-distance test.
    f=friedman.copy()
    extra=pd.DataFrame([{
        "metric":"nearest_stable_abs_state_distance",
        "n_complete_mice":state_summary["n_complete_mice"],
        "friedman_chi2":state_summary["friedman_chi2"],
        "friedman_p":state_summary["friedman_p"],
    }])
    pd.concat([f,extra],ignore_index=True).to_csv(out/"table3_mouse_level_longitudinal_tests.csv",index=False)

    # Table 4: only Holm-significant pairwise results plus all state-distance comparisons.
    sig=pairwise[pairwise["p_holm"]<0.05].copy()
    state_all=state_pair.copy()
    allp=pd.concat([sig,state_all],ignore_index=True,sort=False)
    allp.to_csv(out/"table4_pairwise_longitudinal_results.csv",index=False)

    # Table 5: variance partition.
    cols=["outcome","n_chunks_used","mouse_variance_fraction","mouse_session_variance_fraction",
          "residual_chunk_variance_fraction","session_marginal_R2","session_wald_p_supporting"]
    vc[cols].to_csv(out/"table5_hierarchical_variance_partition.csv",index=False)
    gee.to_csv(out/"tableS1_bistability_GEE_supporting.csv",index=False)


def pfmt(x):
    if x < 0.001: return "<0.001"
    return f"{x:.3f}"


def write_results(cohort, long_df, ss, friedman, pairwise, vc, gee, hier, state_pair, state_summary, out):
    n=len(cohort); nvalid=len(long_df)
    nb=int((long_df["recomputed_regime"]=="bistable").sum()); nm=int((long_df["recomputed_regime"]=="monostable").sum())
    frac=nb/nvalid
    f_b=friedman.loc[friedman.metric=="fraction_bistable"].iloc[0]
    # continuous tests dictionary
    ft={r.metric:r for _,r in friedman.iterrows()}
    sp=state_pair.set_index(["session_a","session_b"])
    vmap=vc.set_index("outcome")
    hsum=hier["bistability_GEE"]
    gee_idx=gee.set_index("term")

    def or_line(day):
        key=next(k for k in gee_idx.index if f"[T.{day}]" in k)
        r=gee_idx.loc[key]
        return float(r.odds_ratio), float(r.ci95_or_low), float(r.ci95_or_high), float(r.p_clustered_supporting)

    or1=or_line("day1"); or7=or_line("day7"); or15=or_line("day15")
    session_fracs={s:float((long_df[long_df.session.astype(str)==s].recomputed_regime=="bistable").mean()) for s in SESSION_ORDER}

    txt=f"""# Results — 40-recording longitudinal extension

## Cohort-scale reproducibility of regression-assisted model selection

The frozen regression-assisted model-selection procedure was applied without retuning to 800 non-overlapping 30-s chunks from 40 recordings (10 mice × four longitudinal sessions). The reduced one-population Wilson–Cowan model with the Laplacian coefficient fixed at $D_E=0$ was selected in all {n} chunks. This conclusion was unchanged after restricting the analysis to the {nvalid} chunks that satisfied the prespecified inference-validity and conditioning criteria ({nvalid}/{n}, {100*nvalid/n:.1f}%). In the unrestricted regression, $D_E$ was positive in {int((cohort.full_regression_D_E>0).sum())}/{n} chunks and the 95% temporal-block bootstrap interval lay entirely above zero in {int((cohort.D_boot_ci95_low>0).sum())}/{n}; however, only {int((cohort.nested_logit_improvement_pct>=0.1).sum())}/{n} and {int((cohort.nested_pde_improvement_pct>=0.1).sum())}/{n} chunks reached the prespecified 0.1% held-out improvement threshold for logit and PDE error, respectively. Thus, non-zero point estimates of $D_E$ occurred, but they did not provide sufficiently robust uncertainty, predictive, and smoothing support to justify the full spatial model.

## Model-implied equilibrium and bifurcation structure

Among the {nvalid} inference-valid chunks, {nb} ({100*frac:.1f}%) were classified as bistable and {nm} ({100*nm/nvalid:.1f}%) as monostable under the selected reduced model. The descriptive bistability fractions were {100*session_fracs['day0']:.1f}% at day0, {100*session_fracs['day1']:.1f}% at day1, {100*session_fracs['day7']:.1f}% at day7, and {100*session_fracs['day15']:.1f}% at day15. However, the prespecified mouse-level repeated-measures analysis did not support a common longitudinal change in bistability prevalence (Friedman $\\chi^2(3)={f_b.friedman_chi2:.2f}$, $p={f_b.friedman_p:.3f}$). Likewise, no significant mouse-level session effect was detected for median $w_{{EE}}$ ($p={ft['median_selected_w_EE'].friedman_p:.3f}$), median $h$ ($p={ft['median_selected_h'].friedman_p:.3f}$), distance to the nearest saddle-node fold ($p={ft['median_distance_to_nearest_fold_h'].friedman_p:.3f}$), or distance to the cusp ($p={ft['median_distance_to_cusp_wh'].friedman_p:.3f}$). The corrected nearest-stable equilibrium eigenvalue showed a non-significant trend across sessions ($p={ft['median_nearest_stable_lambda0'].friedman_p:.3f}$).

The supporting mouse-clustered GEE produced a borderline omnibus session association for chunk-level bistability ($\\chi^2(3)={hsum['session_joint_wald_chi2']:.2f}$, $p={hsum['session_joint_p_supporting']:.3f}$). Relative to day0, the estimated odds ratios were {or1[0]:.2f} for day1 (95% CI {or1[1]:.2f}–{or1[2]:.2f}, $p={or1[3]:.3f}$), {or7[0]:.2f} for day7 (95% CI {or7[1]:.2f}–{or7[2]:.2f}, $p={or7[3]:.3f}$), and {or15[0]:.2f} for day15 (95% CI {or15[1]:.2f}–{or15[2]:.2f}, $p={or15[3]:.3f}$). Because the primary mouse-level test was non-significant and none of the individual GEE contrasts excluded an odds ratio of one, these model-based results were treated as supporting rather than confirmatory evidence.

## Longitudinal position of observed activity relative to stable equilibria

In contrast to the inferred parameter and bifurcation metrics, the distance between the chunk-mean observed activity and the nearest stable equilibrium changed across sessions (Friedman $\\chi^2(3)={state_summary['friedman_chi2']:.2f}$, $p={state_summary['friedman_p']:.4f}$). The mouse-level session median decreased from {state_summary['session_medians']['day0']:.5f} at day0 to {state_summary['session_medians']['day1']:.5f} at day1, {state_summary['session_medians']['day7']:.5f} at day7, and {state_summary['session_medians']['day15']:.5f} at day15. Relative to day0, the decrease was significant at day7 (Holm-adjusted $p={sp.loc[('day0','day7'),'p_holm']:.4f}$) and day15 (Holm-adjusted $p={sp.loc[('day0','day15'),'p_holm']:.4f}$), but not at day1 (Holm-adjusted $p={sp.loc[('day0','day1'),'p_holm']:.3f}$). No significant differences were detected among day1, day7, and day15. Because this outcome was introduced during the extended dynamical analysis rather than specified as the primary longitudinal endpoint, it should be interpreted as a secondary/exploratory finding: by days 7 and 15, observed activity was closer to the nearest model-implied stable equilibrium than at baseline.

## Hierarchical decomposition of dynamical heterogeneity

Variance decomposition showed that the dominant structured variation was generally recording-specific at the mouse-session level rather than a stable mouse effect or a common session trajectory. For $w_{{EE}}$, {100*vmap.loc['selected_w_EE','mouse_session_variance_fraction']:.1f}% of variance was attributed to mouse-session differences, compared with {100*vmap.loc['selected_w_EE','mouse_variance_fraction']:.1f}% between mice and {100*vmap.loc['selected_w_EE','residual_chunk_variance_fraction']:.1f}% at the chunk/residual level. For $h$, the corresponding fractions were {100*vmap.loc['selected_h','mouse_session_variance_fraction']:.1f}%, {100*vmap.loc['selected_h','mouse_variance_fraction']:.1f}%, and {100*vmap.loc['selected_h','residual_chunk_variance_fraction']:.1f}%. Distance to the cusp similarly showed {100*vmap.loc['distance_to_cusp_wh','mouse_session_variance_fraction']:.1f}% mouse-session variance. In contrast, the fixed four-level session factor explained only {100*vmap.loc['selected_w_EE','session_marginal_R2']:.1f}% of marginal variance in $w_{{EE}}$, {100*vmap.loc['selected_h','session_marginal_R2']:.1f}% in $h$, and {100*vmap.loc['distance_to_cusp_wh','session_marginal_R2']:.1f}% in cusp distance. These results indicate substantial recording-specific heterogeneity without a consistent day0-to-day15 shift shared across animals.

## Overall interpretation

Taken together, the cohort extension demonstrates strong reproducibility of structural model selection alongside heterogeneous nonlinear dynamics. The data consistently favoured the reduced $D_E=0$ model, yet the inferred reduced systems occupied both monostable and bistable regions of parameter space. The principal source of parameter variability was mouse-session-specific rather than a uniform longitudinal trajectory. The main positive longitudinal signal was a secondary decrease in the distance between observed activity and the nearest stable equilibrium by days 7 and 15. All regime and equilibrium labels refer to the fitted reduced model in normalized-fluorescence coordinates and should not be interpreted as direct measurements of physiological cortical states.
"""
    (out/"RESULTS_DRAFT.md").write_text(txt)

    objectives="""# Revised objectives for the 40-recording extension

1. **Reproducibility of structural model selection.** Evaluate whether the frozen regression-assisted selector reproducibly supports the full spatial Wilson–Cowan model or the reduced $D_E=0$ model across 40 longitudinal recordings from 10 mice.

2. **Nonlinear dynamical characterization.** Characterize the equilibrium structure, local stability, model-implied mono-/bistability, and proximity to saddle-node bifurcation boundaries of the selected reduced models across inference-valid temporal chunks.

3. **Longitudinal and hierarchical organization.** Test whether bifurcation regime, inferred parameters, and state-to-attractor relationships change systematically across day0, day1, day7, and day15 at the mouse level, and quantify how variability is partitioned across mice, mouse-sessions, and chunks.
"""
    (out/"REVISED_OBJECTIVES.md").write_text(objectives)

    captions="""# Suggested figure captions

**Figure 1. Prespecified model-selection criteria across the 40-recording cohort.** Percentage of the 800 temporal chunks satisfying each frozen regression-assisted support criterion. The full model was selected only when all criteria were satisfied; no chunk met the complete selection rule.

**Figure 2. Inference-valid reduced models in Wilson–Cowan bifurcation parameter space.** Selected $(w_{EE},h)$ parameter pairs for 738 inference-valid chunks are shown relative to the analytic saddle-node fold boundaries and cusp at $(4,-2)$. Points are labelled by the model-implied monostable or bistable regime. Regime labels refer to normalized-fluorescence model coordinates rather than direct physiological states.

**Figure 3. Mouse-level longitudinal prevalence of model-implied bistability.** Lines connect the fraction of inference-valid chunks classified as bistable within each mouse and session; the thicker trajectory denotes the cohort median. Descriptive session differences were not supported by the primary mouse-level Friedman test.

**Figure 4. Longitudinal distance from observed activity to the nearest stable equilibrium.** Mouse-session medians of the absolute distance between chunk-mean normalized activity and the nearest stable equilibrium. The overall session effect was significant, with Holm-corrected paired reductions from day0 to day7 and day15. This was a secondary/exploratory dynamical endpoint.

**Figure 5. Hierarchical variance decomposition of reduced-model dynamical quantities.** Variance fractions from intercept-only mixed models partition variability into mouse, nested mouse-session, and chunk/residual components. For several parameter and bifurcation measures, mouse-session variability dominated while the common fixed session factor explained only a small fraction of marginal variance.
"""
    (out/"FIGURE_CAPTIONS.md").write_text(captions)


def main():
    a=parse_args()
    paths=[a.cohort,a.dynamics,a.longitudinal_chunks,a.subject_session,a.friedman,a.wilcoxon,a.variance,a.gee,a.hier_summary]
    for p in paths: require(p)
    out=a.output_dir; out.mkdir(parents=True,exist_ok=True)

    cohort=pd.read_csv(a.cohort)
    dynamics=pd.read_csv(a.dynamics)
    long_df=pd.read_csv(a.longitudinal_chunks)
    ss=pd.read_csv(a.subject_session)
    friedman=pd.read_csv(a.friedman)
    pairwise=pd.read_csv(a.wilcoxon)
    vc=pd.read_csv(a.variance)
    gee=pd.read_csv(a.gee)
    hier=json.loads(a.hier_summary.read_text())

    if len(cohort)!=800: raise ValueError(f"Expected 800 cohort rows, found {len(cohort)}")
    if len(long_df)!=738: raise ValueError(f"Expected 738 inference-valid longitudinal rows, found {len(long_df)}")
    if cohort["selected_model"].eq("reduced").sum()!=800:
        raise ValueError("Expected 800/800 reduced-model selections; inputs differ from validated cohort result.")

    complete, state_pair, state_summary=state_distance_mouse_tests(long_df)
    state_pair.to_csv(out/"state_distance_pairwise_holm.csv",index=False)
    (out/"state_distance_friedman_summary.json").write_text(json.dumps(state_summary,indent=2))

    compact_tables(cohort,long_df,ss,friedman,pairwise,vc,gee,state_pair,state_summary,out)
    figure_criterion_rates(cohort,out)
    figure_parameter_plane(long_df,out)
    frow=friedman[friedman.metric=="fraction_bistable"].iloc[0]
    figure_spaghetti(ss,"fraction_bistable","Fraction bistable",
                     "Mouse-level longitudinal model-implied bistability",
                     out/"figure3_bistability_longitudinal",
                     annotate=f"Friedman p = {frow.friedman_p:.3f}")

    # build mouse-session medians for state distance
    state_ss=(long_df.groupby(["subject","session"],observed=True)["nearest_stable_abs_state_distance"]
              .median().reset_index())
    figure_spaghetti(state_ss,"nearest_stable_abs_state_distance",
                     "Median |state mean − nearest stable equilibrium|",
                     "Observed activity relative to the nearest stable equilibrium",
                     out/"figure4_state_to_stable_equilibrium_longitudinal",
                     annotate=(f"Friedman p = {state_summary['friedman_p']:.4f}\n"
                               f"day0→day7 Holm p = {state_pair.set_index(['session_a','session_b']).loc[('day0','day7'),'p_holm']:.4f}\n"
                               f"day0→day15 Holm p = {state_pair.set_index(['session_a','session_b']).loc[('day0','day15'),'p_holm']:.4f}"))
    figure_variance_partition(vc,out)

    # supplementary fold proximity
    figure_spaghetti(ss,"median_distance_to_nearest_fold_h",
                     "Median distance to nearest fold in h",
                     "Mouse-level longitudinal proximity to saddle-node boundary",
                     out/"figureS1_fold_distance_longitudinal")

    write_results(cohort,long_df,ss,friedman,pairwise,vc,gee,hier,state_pair,state_summary,out)

    summary={
        "n_chunks_total":800,"n_inference_valid":738,
        "n_reduced_selected":800,
        "n_bistable_valid":int((long_df.recomputed_regime=="bistable").sum()),
        "fraction_bistable_valid":float((long_df.recomputed_regime=="bistable").mean()),
        "state_distance_friedman":state_summary,
        "primary_bistability_friedman_p":float(friedman.loc[friedman.metric=="fraction_bistable","friedman_p"].iloc[0]),
        "supporting_bistability_GEE_joint_p":float(hier["bistability_GEE"]["session_joint_p_supporting"]),
        "guardrail":"Mouse is the biological replication unit; chunk rows are repeated observations.",
    }
    (out/"MANUSCRIPT_SYNTHESIS_SUMMARY.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    print(f"\nManuscript-ready outputs saved to: {out}")

if __name__=="__main__":
    main()
