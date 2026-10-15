#!/usr/bin/env python3
"""
Dynamical stability and bifurcation analysis for cohort-scale selected
single-population Wilson-Cowan models.

Input
-----
outputs/cohort_model_selection_40/cohort_model_selection_800.csv

For every chunk-level selected parameter set (w_EE, D_E, h), this script:
  1. solves the spatially uniform equilibrium equation
         E* = sigmoid(w_EE E* + h),
  2. classifies each equilibrium by local linear stability,
  3. classifies the parameter set as monostable/bistable/degenerate,
  4. computes the one-species reaction-diffusion dispersion relation
         lambda(k) = [-1 + E*(1-E)*(w_EE - D_E k^2)] / tau,
     evaluated on the equilibrium nearest the chunk's normalized state mean,
  5. computes the analytic saddle-node fold boundaries in the (w_EE, h)
     plane when w_EE > 4,
  6. quantifies proximity to the nearest fold and to the cusp (w=4, h=-2),
  7. aggregates descriptive results by recording, subject, and session.

Scientific scope
----------------
The present one-population model can support mono- and bistability through
saddle-node bifurcations. It does not support a classical local Hopf limit
cycle without another dynamical degree of freedom (e.g. inhibition,
adaptation, or delay), so this script deliberately does NOT label an
"oscillatory regime".

D_E is an effective Laplacian spatial-coupling coefficient. For D_E >= 0 in a
one-species model, lambda(k) decreases with k^2; diffusion cannot create a
Turing instability that is absent at k=0. The dispersion calculation is still
useful for quantifying spatial damping where the full model is selected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SESSION_ORDER = ["day0", "day1", "day7", "day15"]
EXPECTED_ROWS = 800


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/cohort_model_selection_40/cohort_model_selection_800.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cohort_dynamics_40"),
    )
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--l-fov", type=float, default=10.0)
    p.add_argument("--grid-n", type=int, default=64)
    p.add_argument("--allow-incomplete", action="store_true")
    return p.parse_args()


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def logit(e):
    e = np.asarray(e, dtype=float)
    return np.log(e / (1.0 - e))


def equilibrium_function(e, w, h):
    # Equivalent to E = sigmoid(w E + h), but numerically convenient on (0,1).
    return float(logit(e) - w * e - h)


def fold_points(w):
    """Return analytic saddle-node fold points for fixed w, or None if w <= 4."""
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


def find_equilibria(w, h, tol=1e-10):
    eps = 1e-10
    cuts = [eps]
    folds = fold_points(w)
    if folds is not None:
        # Critical E positions are useful interval boundaries even when the
        # supplied h is not exactly on a fold.
        ecrit = sorted([folds[0][0], folds[1][0]])
        cuts.extend(ecrit)
    cuts.append(1.0 - eps)
    cuts = sorted(set(cuts))

    roots = []
    # First capture roots at critical points (important very near a fold).
    for x in cuts[1:-1]:
        fx = equilibrium_function(x, w, h)
        if abs(fx) < 1e-8:
            roots.append(float(x))

    # On each interval F is monotone when cut at its critical points.
    for a, b in zip(cuts[:-1], cuts[1:]):
        aa = a + 1e-9 if a > eps else a
        bb = b - 1e-9 if b < 1.0 - eps else b
        if aa >= bb:
            continue
        fa = equilibrium_function(aa, w, h)
        fb = equilibrium_function(bb, w, h)
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0.0:
            roots.append(float(brentq(equilibrium_function, aa, bb, args=(w, h), xtol=tol)))

    roots = sorted(roots)
    unique = []
    for r in roots:
        if not unique or abs(r - unique[-1]) > 1e-7:
            unique.append(r)
    return unique


def local_lambda(e, w, tau):
    sprime = e * (1.0 - e)  # logistic derivative evaluated at equilibrium
    return float((-1.0 + w * sprime) / tau)


def classify_equilibria(roots, w, tau):
    stabilities = []
    for e in roots:
        lam = local_lambda(e, w, tau)
        if lam < -1e-8:
            label = "stable"
        elif lam > 1e-8:
            label = "unstable"
        else:
            label = "neutral"
        stabilities.append((e, lam, label))

    n_stable = sum(s[2] == "stable" for s in stabilities)
    if len(roots) == 1 and n_stable == 1:
        regime = "monostable"
    elif len(roots) == 3 and n_stable == 2:
        regime = "bistable"
    elif any(s[2] == "neutral" for s in stabilities):
        regime = "near_fold"
    else:
        regime = "other"
    return stabilities, regime


def fold_metrics(w, h):
    folds = fold_points(w)
    cusp_distance = float(np.sqrt((w - 4.0) ** 2 + (h + 2.0) ** 2))
    if folds is None:
        return {
            "fold_exists_at_w": False,
            "fold_h_low": np.nan,
            "fold_h_high": np.nan,
            "inside_bistable_wedge": False,
            "distance_to_nearest_fold_h": np.nan,
            "distance_to_cusp_wh": cusp_distance,
        }
    hs = sorted([folds[0][1], folds[1][1]])
    return {
        "fold_exists_at_w": True,
        "fold_h_low": float(hs[0]),
        "fold_h_high": float(hs[1]),
        "inside_bistable_wedge": bool(hs[0] < h < hs[1]),
        "distance_to_nearest_fold_h": float(min(abs(h - hs[0]), abs(h - hs[1]))),
        "distance_to_cusp_wh": cusp_distance,
    }


def dispersion_metrics(e, w, D, tau, l_fov, grid_n):
    dx = l_fov / (grid_n - 1)
    k_nyquist = np.pi / dx
    sprime = e * (1.0 - e)
    lam0 = (-1.0 + sprime * w) / tau
    lam_ny = (-1.0 + sprime * (w - D * k_nyquist ** 2)) / tau
    slope_k2 = -(sprime * D) / tau
    return {
        "lambda_k0": float(lam0),
        "lambda_k_nyquist": float(lam_ny),
        "dispersion_slope_per_k2": float(slope_k2),
        "k_nyquist_rad_per_mm": float(k_nyquist),
        "spatial_modes_more_damped_than_k0": bool(D > 0.0),
    }


def flatten_roots(stabilities):
    out = {}
    for i in range(3):
        if i < len(stabilities):
            e, lam, lab = stabilities[i]
            out[f"equilibrium_{i+1}"] = float(e)
            out[f"equilibrium_{i+1}_lambda0"] = float(lam)
            out[f"equilibrium_{i+1}_stability"] = lab
        else:
            out[f"equilibrium_{i+1}"] = np.nan
            out[f"equilibrium_{i+1}_lambda0"] = np.nan
            out[f"equilibrium_{i+1}_stability"] = ""
    return out


def group_summary(df, group_cols):
    rows = []
    for keys, g in df.groupby(group_cols, sort=True, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update({
            "n_chunks": int(len(g)),
            "fraction_monostable": float((g["dynamical_regime"] == "monostable").mean()),
            "fraction_bistable": float((g["dynamical_regime"] == "bistable").mean()),
            "fraction_near_fold": float((g["dynamical_regime"] == "near_fold").mean()),
            "mean_lambda_nearest_equilibrium": float(g["nearest_equilibrium_lambda0"].mean()),
            "max_lambda_nearest_equilibrium": float(g["nearest_equilibrium_lambda0"].max()),
            "mean_distance_to_cusp_wh": float(g["distance_to_cusp_wh"].mean()),
            "median_distance_to_cusp_wh": float(g["distance_to_cusp_wh"].median()),
            "fraction_full_selected": float((g["selected_model"] == "full").mean()),
            "mean_selected_D_E": float(g["selected_D_E"].mean()),
        })
        finite_fold = g["distance_to_nearest_fold_h"].dropna()
        row["median_distance_to_nearest_fold_h_when_defined"] = (
            float(finite_fold.median()) if len(finite_fold) else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_fold_curve(w_max):
    ws = np.linspace(4.0001, max(4.1, w_max), 1000)
    rows = []
    for w in ws:
        folds = fold_points(float(w))
        if folds is None:
            continue
        for branch, (e, h) in zip(("lower_h", "upper_h"), folds):
            rows.append({"w_EE": w, "h": h, "E_fold": e, "branch": branch})
    return pd.DataFrame(rows)


def make_plots(df, fold_curve, out):
    # 1) Parameter plane with analytic folds.
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    for branch, g in fold_curve.groupby("branch"):
        ax.plot(g["w_EE"], g["h"], linewidth=1.4, label=f"saddle-node fold ({branch})")
    session_markers = {"day0": "o", "day1": "s", "day7": "^", "day15": "D"}
    for sess in SESSION_ORDER:
        g = df[df["session"].astype(str) == sess]
        if len(g):
            ax.scatter(g["selected_w_EE"], g["selected_h"], s=13, alpha=0.55,
                       marker=session_markers[sess], label=sess)
    ax.scatter([4.0], [-2.0], marker="*", s=100, label="cusp (4, -2)")
    ax.set_xlabel(r"$w_{EE}$")
    ax.set_ylabel(r"$h$")
    ax.set_title("Inferred parameter sets relative to saddle-node bifurcation boundaries")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "bifurcation_parameter_plane.png", dpi=240)
    plt.close(fig)

    # 2) Per-recording bistability fraction heatmap.
    heat = (
        df.assign(is_bistable=(df["dynamical_regime"] == "bistable").astype(float))
        .groupby(["subject", "session"], observed=True)["is_bistable"].mean()
        .unstack("session")
        .reindex(columns=SESSION_ORDER)
    )
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    im = ax.imshow(heat.to_numpy(float), aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_xlabel("Session")
    ax.set_ylabel("Mouse")
    ax.set_title("Fraction of chunks classified as bistable")
    fig.colorbar(im, ax=ax, label="Fraction bistable")
    fig.tight_layout()
    fig.savefig(out / "bistability_fraction_subject_session.png", dpi=240)
    plt.close(fig)

    # 3) Stability margin by longitudinal session.
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    vals = [
        df.loc[df["session"].astype(str) == sess, "nearest_equilibrium_lambda0"].dropna().to_numpy()
        for sess in SESSION_ORDER
    ]
    ax.boxplot(vals, showfliers=False)
    ax.set_xticks(range(1, len(SESSION_ORDER) + 1))
    ax.set_xticklabels(SESSION_ORDER)
    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.set_xlabel("Session")
    ax.set_ylabel(r"Local eigenvalue $\lambda(0)$")
    ax.set_title("Local stability margin of equilibrium nearest chunk mean")
    fig.tight_layout()
    fig.savefig(out / "stability_margin_by_session.png", dpi=240)
    plt.close(fig)


def main():
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    required = {
        "task_id", "subject", "session", "chunk_idx", "selected_model",
        "selected_w_EE", "selected_D_E", "selected_h", "state_mean_scaled",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input missing required columns: {sorted(missing)}")
    if len(df) != EXPECTED_ROWS and not args.allow_incomplete:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(df)}")

    rows = []
    for _, r in df.iterrows():
        w = float(r["selected_w_EE"])
        D = float(r["selected_D_E"])
        h = float(r["selected_h"])
        roots = find_equilibria(w, h)
        stabilities, regime = classify_equilibria(roots, w, args.tau)
        if not stabilities:
            raise RuntimeError(
                f"No equilibrium found for {r['subject']} {r['session']} chunk {r['chunk_idx']}"
            )
        state_mean = float(r["state_mean_scaled"])
        nearest_idx = int(np.argmin([abs(e - state_mean) for e, _, _ in stabilities]))
        near_e, near_lam, near_lab = stabilities[nearest_idx]
        fm = fold_metrics(w, h)
        dm = dispersion_metrics(near_e, w, D, args.tau, args.l_fov, args.grid_n)

        row = r.to_dict()
        row.update({
            "n_equilibria": int(len(stabilities)),
            "n_stable_equilibria": int(sum(s[2] == "stable" for s in stabilities)),
            "dynamical_regime": regime,
            "nearest_equilibrium_to_state_mean": float(near_e),
            "nearest_equilibrium_lambda0": float(near_lam),
            "nearest_equilibrium_stability": near_lab,
            "state_mean_minus_nearest_equilibrium": float(state_mean - near_e),
        })
        row.update(flatten_roots(stabilities))
        row.update(fm)
        row.update(dm)
        rows.append(row)

    dyn = pd.DataFrame(rows)
    dyn.to_csv(out / "cohort_dynamics_800.csv", index=False)

    recording = group_summary(dyn, ["task_id", "subject", "session"])
    subject = group_summary(dyn, ["subject"])
    session = group_summary(dyn, ["session"])
    session["session"] = pd.Categorical(session["session"], SESSION_ORDER, ordered=True)
    session = session.sort_values("session")
    recording.to_csv(out / "recording_dynamics_summary_40.csv", index=False)
    subject.to_csv(out / "subject_dynamics_summary_10.csv", index=False)
    session.to_csv(out / "session_dynamics_summary_4.csv", index=False)

    fold_curve = build_fold_curve(max(6.0, float(dyn["selected_w_EE"].max()) + 0.25))
    fold_curve.to_csv(out / "analytic_saddle_node_fold_curve.csv", index=False)
    make_plots(dyn, fold_curve, out)

    summary = {
        "n_rows": int(len(dyn)),
        "n_monostable": int((dyn["dynamical_regime"] == "monostable").sum()),
        "n_bistable": int((dyn["dynamical_regime"] == "bistable").sum()),
        "n_near_fold": int((dyn["dynamical_regime"] == "near_fold").sum()),
        "n_other": int((dyn["dynamical_regime"] == "other").sum()),
        "fraction_bistable_descriptive": float((dyn["dynamical_regime"] == "bistable").mean()),
        "n_full_selected": int((dyn["selected_model"] == "full").sum()),
        "dispersion_interpretation": (
            "For the selected one-population model with D_E >= 0, lambda(k) is non-increasing "
            "with k^2. A positive D_E spatially damps higher-wavenumber perturbations; it does "
            "not create a one-species Turing instability."
        ),
        "bifurcation_interpretation": (
            "Regime labels describe the reduced model in normalized-fluorescence coordinates. "
            "They should not be interpreted as direct measurements of physiological cortical states."
        ),
        "replication_guardrail": (
            "Chunk-level regime counts are repeated observations nested within sessions and mice, "
            "not independent biological replicates."
        ),
    }
    (out / "cohort_dynamics_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nOutputs saved to {out}")


if __name__ == "__main__":
    main()
