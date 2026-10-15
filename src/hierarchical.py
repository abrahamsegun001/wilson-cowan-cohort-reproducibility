#!/usr/bin/env python3
"""
Phase 3C: hierarchical variance decomposition of longitudinal Wilson-Cowan dynamics.

Purpose
-------
Quantify whether variation in the inferred reduced-model dynamics is dominated by
stable between-mouse differences, mouse-by-session differences, or chunk-level
variation, while treating session as a repeated fixed factor.

Input
-----
The corrected Phase 3B chunk table:
  outputs/cohort_longitudinal_40_primary/chunk_dynamics_nearest_stable_equilibrium.csv

Formal structure
----------------
Chunks are nested in mouse-session, which is nested in mouse.

Continuous outcomes:
  y_ijk = fixed(session_j) + u_mouse_i + v_mouse:session_ij + e_ijk

For each outcome, the script fits:
  (a) an intercept-only REML model for variance partitioning; and
  (b) a session-fixed mixed model for descriptive/model-based session effects.

Binary model-implied bistability:
  - GEE binomial model clustered by mouse, with bias-reduced covariance, for
    session effect estimates;
  - Bayesian binomial mixed model (statsmodels BinomialBayesMixedGLM) with
    mouse and mouse-session random intercepts for latent-scale variance/ICC.

Important inferential guardrail
-------------------------------
The mouse-level Friedman tests from Phase 3B remain the primary confirmatory
session tests because there are only 10 biological subjects. Mixed-model and GEE
session p-values here are supporting/model-based analyses. The primary purpose
of this script is variance decomposition and heterogeneity quantification.

"Bistable" is a property of the fitted reduced model in normalized-fluorescence
coordinates; it is not a direct physiological state label.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
except ImportError as exc:
    raise SystemExit(
        "This analysis requires statsmodels. Activate gpinn_env and install/load "
        "statsmodels before rerunning. Original error: " + str(exc)
    )

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SESSION_ORDER = ["day0", "day1", "day7", "day15"]
EXPECTED_SUBJECTS = 10
EXPECTED_SUBJECT_SESSIONS = 40

CONTINUOUS_OUTCOMES = {
    "selected_w_EE": "Selected w_EE",
    "selected_h": "Selected h",
    "distance_to_cusp_wh": "Distance to cusp",
    "nearest_stable_lambda0": "Nearest-stable lambda0",
    "nearest_stable_abs_state_distance": "State distance to nearest stable equilibrium",
    "distance_to_nearest_fold_h": "Distance to nearest fold in h",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/cohort_longitudinal_40_primary/"
            "chunk_dynamics_nearest_stable_equilibrium.csv"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cohort_hierarchical_40_primary"),
    )
    return p.parse_args()


def prepare_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "subject", "session", "chunk_idx", "selected_w_EE", "selected_h",
        "distance_to_cusp_wh", "distance_to_nearest_fold_h",
        "nearest_stable_lambda0", "nearest_stable_abs_state_distance",
        "recomputed_regime",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input missing required columns: {sorted(missing)}")

    df = df.copy()
    df["subject"] = df["subject"].astype(str)
    df["session"] = pd.Categorical(df["session"].astype(str), SESSION_ORDER, ordered=True)
    df["subject_session"] = df["subject"] + ":" + df["session"].astype(str)
    df["is_bistable"] = (df["recomputed_regime"].astype(str) == "bistable").astype(int)

    if df["subject"].nunique() != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Expected {EXPECTED_SUBJECTS} subjects, found {df['subject'].nunique()}"
        )
    if df["subject_session"].nunique() != EXPECTED_SUBJECT_SESSIONS:
        raise ValueError(
            f"Expected {EXPECTED_SUBJECT_SESSIONS} subject-sessions, "
            f"found {df['subject_session'].nunique()}"
        )
    if set(df["session"].dropna().astype(str).unique()) != set(SESSION_ORDER):
        raise ValueError("Unexpected/missing session labels")
    return df


def fit_mixedlm_with_fallback(formula: str, data: pd.DataFrame, reml: bool):
    model = smf.mixedlm(
        formula,
        data=data,
        groups=data["subject"],
        re_formula="1",
        vc_formula={"mouse_session": "0 + C(subject_session)"},
    )
    errors = []
    for method in ("lbfgs", "bfgs", "powell"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(
                    reml=reml,
                    method=method,
                    maxiter=3000,
                    disp=False,
                )
            if np.isfinite(result.llf):
                return result, method, errors
        except Exception as exc:
            errors.append(f"{method}: {type(exc).__name__}: {exc}")
    raise RuntimeError("MixedLM failed; " + " | ".join(errors))


def variance_components_from_result(res):
    mouse_var = float(res.cov_re.iloc[0, 0]) if res.cov_re.size else 0.0
    mouse_session_var = float(res.vcomp[0]) if len(res.vcomp) else 0.0
    residual_var = float(res.scale)
    total = mouse_var + mouse_session_var + residual_var
    return {
        "mouse_variance": mouse_var,
        "mouse_session_variance": mouse_session_var,
        "residual_chunk_variance": residual_var,
        "total_random_residual_variance": total,
        "mouse_variance_fraction": mouse_var / total if total > 0 else np.nan,
        "mouse_session_variance_fraction": mouse_session_var / total if total > 0 else np.nan,
        "residual_chunk_variance_fraction": residual_var / total if total > 0 else np.nan,
        "icc_same_mouse_different_session": mouse_var / total if total > 0 else np.nan,
        "icc_same_mouse_same_session": (
            (mouse_var + mouse_session_var) / total if total > 0 else np.nan
        ),
    }


def session_fixed_means(res) -> dict[str, float]:
    params = res.fe_params
    intercept = float(params.get("Intercept", 0.0))
    out = {"day0": intercept}
    for s in SESSION_ORDER[1:]:
        target = None
        for name in params.index:
            if name != "Intercept" and f"[T.{s}]" in name:
                target = name
                break
        out[s] = intercept + (float(params[target]) if target is not None else 0.0)
    return out


def joint_session_wald(res):
    names = list(res.fe_params.index)
    session_names = [n for n in names if n != "Intercept" and "C(session" in n]
    if not session_names:
        return np.nan, 0, np.nan
    try:
        beta = res.fe_params.loc[session_names].to_numpy(float)
        cov_all = res.cov_params()
        cov = cov_all.loc[session_names, session_names].to_numpy(float)
        stat = float(beta.T @ np.linalg.pinv(cov) @ beta)
        df_wald = len(session_names)
        p = float(chi2.sf(stat, df_wald))
        return stat, df_wald, p
    except Exception:
        return np.nan, len(session_names), np.nan


def fit_continuous_outcome(df: pd.DataFrame, outcome: str):
    dat = df[["subject", "session", "subject_session", outcome]].dropna().copy()
    if dat[outcome].nunique() < 2:
        raise ValueError(f"{outcome} has insufficient variation")

    null_res, null_method, _ = fit_mixedlm_with_fallback(f"{outcome} ~ 1", dat, reml=True)
    vc = variance_components_from_result(null_res)

    fixed_formula = (
        f"{outcome} ~ C(session, Treatment(reference='day0'))"
    )
    fixed_res, fixed_method, _ = fit_mixedlm_with_fallback(fixed_formula, dat, reml=True)
    fixed_vc = variance_components_from_result(fixed_res)

    means = session_fixed_means(fixed_res)
    fixed_var = float(np.var([means[s] for s in SESSION_ORDER], ddof=0))
    denom = (
        fixed_var
        + fixed_vc["mouse_variance"]
        + fixed_vc["mouse_session_variance"]
        + fixed_vc["residual_chunk_variance"]
    )
    marginal_r2 = fixed_var / denom if denom > 0 else np.nan
    conditional_r2 = (
        fixed_var + fixed_vc["mouse_variance"] + fixed_vc["mouse_session_variance"]
    ) / denom if denom > 0 else np.nan
    stat, df_wald, p_wald = joint_session_wald(fixed_res)

    summary = {
        "outcome": outcome,
        "n_chunks_used": int(len(dat)),
        "n_mice": int(dat["subject"].nunique()),
        "n_mouse_sessions": int(dat["subject_session"].nunique()),
        "null_optimizer": null_method,
        **vc,
        "session_fixed_variance_equal_session_weight": fixed_var,
        "session_marginal_R2": marginal_r2,
        "conditional_R2_session_plus_random_intercepts": conditional_r2,
        "session_wald_chi2": stat,
        "session_wald_df": df_wald,
        "session_wald_p_supporting": p_wald,
    }

    effects = []
    ci = fixed_res.conf_int()
    for name in fixed_res.fe_params.index:
        est = float(fixed_res.fe_params[name])
        se = float(fixed_res.bse_fe[name])
        z = est / se if se > 0 else np.nan
        p = 2.0 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
        effects.append({
            "outcome": outcome,
            "term": name,
            "estimate": est,
            "se": se,
            "z": z,
            "p_model_based": p,
            "ci95_low": float(ci.loc[name, 0]),
            "ci95_high": float(ci.loc[name, 1]),
        })
    return summary, effects


def fit_bistability_gee(df: pd.DataFrame):
    dat = df[["subject", "session", "is_bistable"]].dropna().copy()
    formula = "is_bistable ~ C(session, Treatment(reference='day0'))"
    model = smf.gee(
        formula,
        groups="subject",
        data=dat,
        family=sm.families.Binomial(),
        cov_struct=Exchangeable(),
    )
    try:
        res = model.fit(cov_type="bias_reduced", maxiter=1000)
        cov_used = "bias_reduced"
    except Exception:
        res = model.fit(cov_type="robust", maxiter=1000)
        cov_used = "robust"

    rows = []
    ci = res.conf_int()
    for name in res.params.index:
        beta = float(res.params[name])
        se = float(res.bse[name])
        rows.append({
            "term": name,
            "log_odds": beta,
            "se": se,
            "odds_ratio": float(np.exp(beta)),
            "ci95_or_low": float(np.exp(ci.loc[name, 0])),
            "ci95_or_high": float(np.exp(ci.loc[name, 1])),
            "p_clustered_supporting": float(res.pvalues[name]),
            "covariance": cov_used,
        })

    names = list(res.params.index)
    idx = [i for i, n in enumerate(names) if n != "Intercept" and "C(session" in n]
    if idx:
        R = np.zeros((len(idx), len(names)))
        for r, i in enumerate(idx):
            R[r, i] = 1.0
        wt = res.wald_test(R, scalar=True)
        stat = float(np.asarray(wt.statistic).squeeze())
        p_joint = float(wt.pvalue)
        df_joint = len(idx)
    else:
        stat, p_joint, df_joint = np.nan, np.nan, 0

    summary = {
        "n_chunks": int(len(dat)),
        "n_mice": int(dat["subject"].nunique()),
        "covariance": cov_used,
        "working_correlation_alpha": (
            float(np.asarray(res.cov_struct.dep_params).squeeze())
            if res.cov_struct.dep_params is not None else None
        ),
        "session_joint_wald_chi2": stat,
        "session_joint_df": df_joint,
        "session_joint_p_supporting": p_joint,
    }
    return summary, rows


def fit_bistability_random_effects(df: pd.DataFrame):
    dat = df[["subject", "session", "subject_session", "is_bistable"]].dropna().copy()
    formula = "is_bistable ~ C(session, Treatment(reference='day0'))"
    vc = {
        "mouse": "0 + C(subject)",
        "mouse_session": "0 + C(subject_session)",
    }
    model = BinomialBayesMixedGLM.from_formula(formula, vc, dat)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit_vb()

    vcp_names = list(model.vcp_names)
    log_sd = dict(zip(vcp_names, np.asarray(res.vcp_mean, dtype=float)))
    sd = {k: float(np.exp(v)) for k, v in log_sd.items()}
    var = {k: float(sd[k] ** 2) for k in sd}
    logistic_residual = math.pi ** 2 / 3.0
    mouse_var = var.get("mouse", np.nan)
    mouse_session_var = var.get("mouse_session", np.nan)
    total = mouse_var + mouse_session_var + logistic_residual

    fe_names = list(model.exog_names)
    fe_mean = np.asarray(res.fe_mean, dtype=float)
    fe_sd = np.asarray(res.fe_sd, dtype=float)
    fixed = []
    for name, est, se in zip(fe_names, fe_mean, fe_sd):
        z = est / se if se > 0 else np.nan
        fixed.append({
            "term": name,
            "posterior_mean_log_odds": float(est),
            "posterior_sd": float(se),
            "approx_z": float(z),
            "approx_p": float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan,
            "odds_ratio": float(np.exp(est)),
        })

    # Equal weighting over the four fixed session logits.
    params = dict(zip(fe_names, fe_mean))
    intercept = float(params.get("Intercept", 0.0))
    fixed_logits = [intercept]
    for s in SESSION_ORDER[1:]:
        match = next((n for n in fe_names if f"[T.{s}]" in n), None)
        fixed_logits.append(intercept + (float(params[match]) if match else 0.0))
    session_fixed_logit_var = float(np.var(fixed_logits, ddof=0))
    denom = session_fixed_logit_var + total

    summary = {
        "method": "statsmodels BinomialBayesMixedGLM variational Bayes",
        "n_chunks": int(len(dat)),
        "mouse_log_sd": log_sd.get("mouse"),
        "mouse_session_log_sd": log_sd.get("mouse_session"),
        "mouse_sd_logit": sd.get("mouse"),
        "mouse_session_sd_logit": sd.get("mouse_session"),
        "mouse_variance_logit": mouse_var,
        "mouse_session_variance_logit": mouse_session_var,
        "logistic_residual_variance": logistic_residual,
        "latent_mouse_ICC": mouse_var / total if total > 0 else np.nan,
        "latent_same_mouse_session_ICC": (
            (mouse_var + mouse_session_var) / total if total > 0 else np.nan
        ),
        "session_fixed_logit_variance_equal_session_weight": session_fixed_logit_var,
        "session_marginal_R2_latent": session_fixed_logit_var / denom if denom > 0 else np.nan,
        "interpretation": (
            "Random-effect variance estimates are latent-logit-scale descriptive "
            "heterogeneity estimates. GEE/Friedman results should be used for session inference."
        ),
    }
    return summary, fixed


def mouse_bistability_summary(df: pd.DataFrame):
    return (
        df.groupby("subject", observed=True)
        .agg(
            n_chunks=("is_bistable", "size"),
            n_bistable=("is_bistable", "sum"),
            fraction_bistable=("is_bistable", "mean"),
        )
        .reset_index()
        .sort_values("fraction_bistable", ascending=False)
    )


def plot_continuous_variance_components(vc: pd.DataFrame, path: Path):
    plot = vc.set_index("outcome")[[
        "mouse_variance_fraction",
        "mouse_session_variance_fraction",
        "residual_chunk_variance_fraction",
    ]]
    ax = plot.plot(kind="bar", stacked=True, figsize=(10, 5.5))
    ax.set_ylabel("Fraction of random/residual variance")
    ax.set_xlabel("Outcome")
    ax.set_ylim(0, 1)
    ax.set_title("Hierarchical variance decomposition")
    ax.legend(["Mouse", "Mouse-session", "Chunk/residual"], fontsize=8)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()


def plot_mouse_bistability(summary: pd.DataFrame, path: Path):
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x, summary["fraction_bistable"].to_numpy(float))
    ax.set_xticks(x)
    ax.set_xticklabels(summary["subject"].astype(str), rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction model-implied bistable")
    ax.set_xlabel("Mouse")
    ax.set_title("Between-mouse heterogeneity in bistability prevalence")
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def clean_json(obj):
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def main():
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    df = prepare_data(args.input)

    vc_rows = []
    effect_rows = []
    failures = []
    for outcome in CONTINUOUS_OUTCOMES:
        try:
            summary, effects = fit_continuous_outcome(df, outcome)
            summary["label"] = CONTINUOUS_OUTCOMES[outcome]
            vc_rows.append(summary)
            effect_rows.extend(effects)
        except Exception as exc:
            failures.append({"outcome": outcome, "error": f"{type(exc).__name__}: {exc}"})

    if not vc_rows:
        raise RuntimeError("All continuous mixed models failed")

    vc_df = pd.DataFrame(vc_rows)
    vc_df.to_csv(out / "continuous_variance_components.csv", index=False)
    pd.DataFrame(effect_rows).to_csv(out / "continuous_session_fixed_effects.csv", index=False)
    pd.DataFrame(failures).to_csv(out / "continuous_model_failures.csv", index=False)

    gee_summary, gee_rows = fit_bistability_gee(df)
    pd.DataFrame(gee_rows).to_csv(out / "bistability_GEE_session_effects.csv", index=False)

    try:
        glmm_summary, glmm_fixed = fit_bistability_random_effects(df)
        glmm_error = None
        pd.DataFrame(glmm_fixed).to_csv(out / "bistability_random_effects_fixed_terms.csv", index=False)
    except Exception as exc:
        glmm_summary = None
        glmm_fixed = []
        glmm_error = f"{type(exc).__name__}: {exc}"

    mouse_b = mouse_bistability_summary(df)
    mouse_b.to_csv(out / "mouse_overall_bistability_10.csv", index=False)

    plot_continuous_variance_components(vc_df, out / "continuous_variance_components.png")
    plot_mouse_bistability(mouse_b, out / "mouse_overall_bistability.png")

    # A compact ranking by between-mouse variance fraction is descriptive only.
    heterogeneity_rank = (
        vc_df[["outcome", "mouse_variance_fraction", "session_marginal_R2"]]
        .sort_values("mouse_variance_fraction", ascending=False)
        .to_dict(orient="records")
    )

    summary = {
        "input_rows": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "n_subject_sessions": int(df["subject_session"].nunique()),
        "continuous_model": (
            "linear mixed model with fixed session, random mouse intercept, "
            "random nested mouse-session intercept, and chunk residual"
        ),
        "continuous_variance_partition": (
            "intercept-only REML model; fractions partition mouse, mouse-session, "
            "and chunk/residual variability"
        ),
        "bistability_GEE": gee_summary,
        "bistability_random_effects": glmm_summary,
        "bistability_random_effects_error": glmm_error,
        "continuous_failures": failures,
        "continuous_heterogeneity_ranking_descriptive": heterogeneity_rank,
        "guardrails": [
            "Mouse is the biological replication unit; chunks are not independent biological replicates.",
            "Phase 3B mouse-level Friedman tests remain the primary session-effect tests with n=10 mice.",
            "Mixed-model/GEE session p-values are supporting model-based analyses.",
            "Variance components quantify heterogeneity and do not establish biological causation.",
            "Bistability is model-implied in normalized-fluorescence coordinates, not a direct physiological state label.",
        ],
    }
    summary = clean_json(summary)
    (out / "hierarchical_variance_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== CONTINUOUS VARIANCE COMPONENTS ===")
    cols = [
        "outcome", "n_chunks_used", "mouse_variance_fraction",
        "mouse_session_variance_fraction", "residual_chunk_variance_fraction",
        "session_marginal_R2", "session_wald_p_supporting",
    ]
    print(vc_df[cols].to_string(index=False))
    print("\n=== BISTABILITY GEE ===")
    print(json.dumps(clean_json(gee_summary), indent=2))
    print("\n=== BISTABILITY RANDOM EFFECTS ===")
    print(json.dumps(clean_json(glmm_summary), indent=2))
    if failures:
        print("\nContinuous model failures:")
        print(pd.DataFrame(failures).to_string(index=False))
    if glmm_error:
        print("\nBistability random-effects model failed:", glmm_error)
    print(f"\nOutputs saved to {out}")


if __name__ == "__main__":
    main()
