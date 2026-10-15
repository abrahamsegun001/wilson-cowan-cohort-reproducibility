#!/usr/bin/env python3
"""Aggregate the 40 independent cohort gPINN fits and build manuscript outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SESSIONS = ["day0", "day1", "day7", "day15"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("outputs/cohort_gpinn_reconstruction_40"))
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def load_all_metrics(root: Path):
    files = sorted((root / "per_recording").glob("*_metrics.csv"))
    if len(files) != 40:
        raise RuntimeError(f"Expected exactly 40 per-recording metric files; found {len(files)}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if len(df) != 40 or df[["subject", "session"]].drop_duplicates().shape[0] != 40:
        raise RuntimeError("Metrics do not contain exactly 40 unique recordings")
    return df.sort_values("task_id").reset_index(drop=True)


def nearest_quantile_row(df: pd.DataFrame, q: float):
    target = float(df["random_val_state_r2"].quantile(q, interpolation="linear"))
    z = df.copy()
    z["_dist"] = (z["random_val_state_r2"] - target).abs()
    return z.sort_values(["_dist", "task_id"]).iloc[0], target


def recon_file(root: Path, row: pd.Series):
    patt = f"task-{int(row.task_id):02d}_{row.subject}_{row.session}_chunk-{int(row.chunk_idx):02d}_reconstruction.npz"
    p = root / "reconstructions" / patt
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def make_figure(df, root: Path, out: Path, med_row: pd.Series):
    rec = np.load(recon_file(root, med_row))
    obs = rec["observed_frames"]
    pred = rec["predicted_frames"]
    err = rec["absolute_error_frames"]
    times = rec["frame_times_s"]
    tr_t = rec["trace_times_s"]
    tr_o = rec["trace_observed_mean"]
    tr_p = rec["trace_predicted_mean"]
    val_o = rec["validation_observed"]
    val_p = rec["validation_predicted"]

    # Display contrast is defined from observed data only, then applied
    # unchanged to both observed and predicted maps. This preserves a fair
    # visual comparison while preventing isolated extrema from flattening
    # anatomical/spatial contrast.
    finite_obs = obs[np.isfinite(obs)]
    vmin, vmax = np.percentile(finite_obs, [1.0, 99.0])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(obs))
        vmax = float(np.nanmax(obs))

    finite_err = err[np.isfinite(err)]
    err_vmax = float(np.percentile(finite_err, 99.0))
    if not np.isfinite(err_vmax) or err_vmax <= 0:
        err_vmax = float(np.nanmax(err))

    # Compact manuscript layout.  The middle reconstruction block is given a
    # fixed, centred footprint so the square maps remain large without the
    # excessive horizontal whitespace produced by a full-width 3-column grid.
    fig = plt.figure(figsize=(13.6, 12.2))
    outer = fig.add_gridspec(
        3, 1,
        height_ratios=[1.12, 3.90, 1.38],
        hspace=0.32,
        left=0.065,
        right=0.965,
        bottom=0.055,
        top=0.972,
    )

    # ------------------------------ top row ------------------------------
    top_grid = outer[0, 0].subgridspec(1, 2, wspace=0.18)

    # A: cohort reconstruction accuracy.
    ax = fig.add_subplot(top_grid[0, 0])
    data = [df.loc[df.session == s, "random_val_state_r2"].to_numpy() for s in SESSIONS]
    ax.boxplot(data, tick_labels=SESSIONS, showfliers=False)
    rng = np.random.default_rng(42)
    for i, vals in enumerate(data, start=1):
        jitter = rng.normal(0, 0.035, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=26, alpha=0.75)
    ax.axhline(
        df["random_val_state_r2"].median(),
        linestyle="--", linewidth=1.2,
        label=f"cohort median = {df['random_val_state_r2'].median():.3f}",
    )
    ax.set_ylabel("Held-out reconstruction $R^2$")
    ax.set_xlabel("Session")
    ax.set_title("A  Cohort reconstruction accuracy", pad=7)
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    # B: reconstruction versus physics residual.
    ax = fig.add_subplot(top_grid[0, 1])
    ax.scatter(
        df["random_val_state_r2"], df["random_val_neural_pde_rmse"],
        s=38, alpha=0.8,
    )
    ax.scatter(
        [med_row.random_val_state_r2], [med_row.random_val_neural_pde_rmse],
        s=105, marker="*", label="median-$R^2$ example",
    )
    ax.set_xlabel("Held-out reconstruction $R^2$")
    ax.set_ylabel("Neural PDE RMSE")
    ax.set_title("B  Reconstruction versus physics residual", pad=7)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # ----------------------- middle reconstruction block -----------------
    # Side spacer columns centre a compact map block instead of stretching
    # the three square map columns across the full figure width.
    mid_wrap = outer[1, 0].subgridspec(
        1, 3,
        width_ratios=[0.72, 2.85, 0.72],
        wspace=0.0,
    )
    maps = mid_wrap[0, 1].subgridspec(
        3, 4,
        width_ratios=[1.0, 1.0, 1.0, 0.040],
        hspace=0.075,
        wspace=0.065,
    )

    map_axes = [[None] * 3 for _ in range(3)]
    row_labels = ["Observed", "gPINN reconstruction", "Absolute error"]
    im_act = None
    im_err = None

    for r in range(3):
        for c in range(3):
            axm = fig.add_subplot(maps[r, c])
            map_axes[r][c] = axm
            if r == 0:
                im_act = axm.imshow(
                    obs[c], vmin=vmin, vmax=vmax,
                    origin="lower", aspect="equal",
                )
                axm.set_title(f"$t={times[c]:.1f}$ s", fontsize=10, pad=4)
            elif r == 1:
                axm.imshow(
                    pred[c], vmin=vmin, vmax=vmax,
                    origin="lower", aspect="equal",
                )
            else:
                im_err = axm.imshow(
                    err[c], vmin=0.0, vmax=err_vmax,
                    origin="lower", aspect="equal",
                )
            axm.set_xticks([])
            axm.set_yticks([])
            if c == 0:
                axm.set_ylabel(row_labels[r], fontsize=10, labelpad=8)

    # Dedicated colorbar cells prevent the bars from changing map spacing.
    cax_activity = fig.add_subplot(maps[0:2, 3])
    cb1 = fig.colorbar(im_act, cax=cax_activity, orientation="vertical")
    cb1.set_label("Normalized activity", fontsize=9)
    cb1.ax.tick_params(labelsize=8)

    cax_error = fig.add_subplot(maps[2, 3])
    cb2 = fig.colorbar(im_err, cax=cax_error, orientation="vertical")
    cb2.set_label("Absolute error", fontsize=9)
    cb2.ax.tick_params(labelsize=8)

    # Group title and panel label.
    left = map_axes[0][0].get_position().x0
    right = map_axes[0][2].get_position().x1
    top = map_axes[0][0].get_position().y1
    fig.text(left - 0.020, top + 0.018, "C", fontsize=12, fontweight="bold", va="bottom")
    fig.text(
        (left + right) / 2,
        top + 0.018,
        f"Median-performing example: {med_row.subject}, {med_row.session}, "
        f"chunk {int(med_row.chunk_idx)} ($R^2={med_row.random_val_state_r2:.3f}$)",
        ha="center", va="bottom", fontsize=10,
    )

    # ----------------------------- bottom row ----------------------------
    bottom_grid = outer[2, 0].subgridspec(
        1, 2,
        width_ratios=[1.15, 2.10],
        wspace=0.22,
    )

    # D: held-out pointwise agreement.
    ax = fig.add_subplot(bottom_grid[0, 0])
    hb = ax.hexbin(val_o, val_p, gridsize=46, mincnt=1, bins="log")
    lo = float(min(val_o.min(), val_p.min()))
    hi = float(max(val_o.max(), val_p.max()))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title("D  Held-out pointwise agreement", pad=7)
    cb = fig.colorbar(hb, ax=ax, shrink=0.84, pad=0.025)
    cb.set_label("log count", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # E: spatially averaged activity across the representative chunk.
    ax = fig.add_subplot(bottom_grid[0, 1])
    ax.plot(tr_t, tr_o, linewidth=1.45, label="Observed")
    ax.plot(tr_t, tr_p, linewidth=1.45, label="gPINN")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spatial mean normalized activity")
    ax.set_title("E  Spatially averaged activity", pad=7)
    ax.legend(
        frameon=False, ncol=2, loc="lower right",
        bbox_to_anchor=(1.0, 1.015), borderaxespad=0.0, fontsize=9,
    )

    for ext in ["png", "pdf"]:
        fig.savefig(
            out / f"figure_gpinn_cohort_reconstruction.{ext}",
            dpi=300 if ext == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)

def main():
    args = parse_args()
    root = args.input_dir
    out = args.output_dir or root
    out.mkdir(parents=True, exist_ok=True)
    df = load_all_metrics(root)
    df.to_csv(out / "cohort_gpinn_reconstruction_metrics_40.csv", index=False)

    q25, q25_target = nearest_quantile_row(df, 0.25)
    med, med_target = nearest_quantile_row(df, 0.50)
    q75, q75_target = nearest_quantile_row(df, 0.75)

    examples = []
    for label, row, target in [("q25", q25, q25_target), ("median", med, med_target), ("q75", q75, q75_target)]:
        examples.append({
            "label": label,
            "target_quantile_r2": float(target),
            "task_id": int(row.task_id),
            "subject": str(row.subject),
            "session": str(row.session),
            "chunk_idx": int(row.chunk_idx),
            "random_val_state_r2": float(row.random_val_state_r2),
            "random_val_state_mse": float(row.random_val_state_mse),
            "random_val_neural_pde_rmse": float(row.random_val_neural_pde_rmse),
        })
    pd.DataFrame(examples).to_csv(out / "representative_examples_by_r2_quantile.csv", index=False)

    by_session = (
        df.groupby("session")
          .agg(n_recordings=("task_id", "size"),
               mean_r2=("random_val_state_r2", "mean"),
               sd_r2=("random_val_state_r2", "std"),
               median_r2=("random_val_state_r2", "median"),
               mean_mse=("random_val_state_mse", "mean"),
               mean_neural_pde_rmse=("random_val_neural_pde_rmse", "mean"))
          .reindex(SESSIONS)
          .reset_index()
    )
    by_session.to_csv(out / "cohort_gpinn_reconstruction_by_session.csv", index=False)

    summary = {
        "n_recordings": 40,
        "selection_rule": "For each recording, use the inference-valid chunk closest to temporal midpoint chunk 10; lower chunk index breaks ties; selection is made before gPINN training and is independent of reconstruction performance.",
        "training_design": "Independent cold-start fixed-parameter gPINN per recording; no sequential warm-start and no continuity loss.",
        "validation_interpretation": "Random-pointwise spatiotemporal interpolation within each selected 30-s chunk; not forecasting.",
        "state_r2": {
            "mean": float(df.random_val_state_r2.mean()),
            "sd": float(df.random_val_state_r2.std(ddof=1)),
            "median": float(df.random_val_state_r2.median()),
            "q25": float(df.random_val_state_r2.quantile(0.25)),
            "q75": float(df.random_val_state_r2.quantile(0.75)),
            "min": float(df.random_val_state_r2.min()),
            "max": float(df.random_val_state_r2.max()),
        },
        "state_mse_mean": float(df.random_val_state_mse.mean()),
        "neural_pde_rmse_mean": float(df.random_val_neural_pde_rmse.mean()),
        "median_performance_example": examples[1],
        "important_guardrail": "The median-performing example is selected transparently after fitting only for visualization; all quantitative claims use all 40 recordings.",
    }
    (out / "cohort_gpinn_reconstruction_summary.json").write_text(json.dumps(summary, indent=2))

    make_figure(df, root, out, med)

    caption = f"""# Cohort gPINN reconstruction figure caption

**Figure X. Fixed-parameter gPINN reconstruction across 40 longitudinal recordings.** One independently initialized gPINN was trained for each recording using a deterministic, reconstruction-independent representative-chunk rule: the inference-valid chunk closest to the temporal midpoint (chunk 10), with lower chunk index breaking ties. Physical coefficients were fixed to the values selected by the frozen regression-assisted cohort analysis; no sequential weight transfer or continuity regularization was used. (A) Held-out random-pointwise reconstruction $R^2$ across sessions. Each point represents one recording (10 mice per session); the dashed line denotes the cohort median. (B) Relationship between held-out reconstruction $R^2$ and neural PDE RMSE across all 40 recordings. The star identifies the recording whose $R^2$ is closest to the cohort median. (C) Observed activity, gPINN reconstruction, and absolute error at the beginning, midpoint, and end of the prespecified 30-s chunk from the median-performing recording ({med.subject}, {med.session}, chunk {int(med.chunk_idx)}; held-out $R^2$={med.random_val_state_r2:.3f}). Observed and reconstructed maps share a single display range derived from the 1st--99th percentiles of the observed representative frames; the same limits are applied to both rows, and no prediction-dependent contrast normalization is used. Absolute-error maps share a common range capped at the 99th percentile for display. (D) Held-out observed-versus-predicted pointwise activity for the same recording. (E) Spatially averaged observed and reconstructed activity across the representative chunk. Validation represents random-pointwise spatiotemporal interpolation within each 30-s chunk and should not be interpreted as future-time forecasting.
"""
    (out / "FIGURE_GPinn_COHORT_CAPTION.md").write_text(caption)

    print(json.dumps(summary, indent=2))
    print(f"Outputs written to {out}")


if __name__ == "__main__":
    main()
