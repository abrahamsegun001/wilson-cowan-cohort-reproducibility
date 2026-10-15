#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    base = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    recon_dir = base / "reconstructions"
    rep = pd.read_csv(base / "representative_examples_by_r2_quantile.csv")

    label_names = {"q25":"25th percentile","median":"Median","q75":"75th percentile"}

    fig = plt.figure(figsize=(15.8, 8.2))
    outer = fig.add_gridspec(
        3, 1, left=0.075, right=0.965, bottom=0.075, top=0.90, hspace=0.23
    )

    fig.text(0.235, 0.945, "Observed", ha="center", va="center",
             fontsize=12.5, fontweight="bold")
    fig.text(0.505, 0.945, "gPINN reconstruction", ha="center", va="center",
             fontsize=12.5, fontweight="bold")
    fig.text(0.775, 0.945, "Absolute error", ha="center", va="center",
             fontsize=12.5, fontweight="bold")

    for bi, row in rep.iterrows():
        task_id = int(row["task_id"])
        subject = str(row["subject"])
        session = str(row["session"])
        chunk = int(row["chunk_idx"])
        f = recon_dir / (
            f"task-{task_id:02d}_{subject}_{session}_chunk-{chunk:02d}_reconstruction.npz"
        )
        dat = np.load(f)
        times = dat["frame_times_s"]
        obs = dat["observed_frames"]
        pred = dat["predicted_frames"]
        err = dat["absolute_error_frames"]

        vmin, vmax = np.percentile(obs, [1, 99])
        emax = np.percentile(err, 99)
        if not np.isfinite(emax) or emax <= 0:
            emax = float(np.nanmax(err))

        sub = outer[bi].subgridspec(
            1, 11,
            width_ratios=[1,1,1,0.06,1,1,1,0.06,1,1,1],
            wspace=0.06
        )

        axes_obs = [fig.add_subplot(sub[0,c]) for c in range(3)]
        axes_pred = [fig.add_subplot(sub[0,c]) for c in range(4,7)]
        axes_err = [fig.add_subplot(sub[0,c]) for c in range(8,11)]

        for c in range(3):
            axes_obs[c].imshow(obs[c], cmap="viridis", vmin=vmin, vmax=vmax,
                               interpolation="nearest")
            axes_pred[c].imshow(pred[c], cmap="viridis", vmin=vmin, vmax=vmax,
                                interpolation="nearest")
            axes_err[c].imshow(err[c], cmap="viridis", vmin=0, vmax=emax,
                               interpolation="nearest")
            for ax in (axes_obs[c], axes_pred[c], axes_err[c]):
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.65)
            if bi == 0:
                axes_obs[c].set_title(rf"$t={times[c]:.0f}$ s", fontsize=9.5, pad=2)
                axes_pred[c].set_title(rf"$t={times[c]:.0f}$ s", fontsize=9.5, pad=2)
                axes_err[c].set_title(rf"$t={times[c]:.0f}$ s", fontsize=9.5, pad=2)

        pos = axes_obs[0].get_position()
        row_text = (
            f"{chr(ord('A')+bi)}  {label_names[row['label']]}\n"
            f"{subject}, {session}, chunk {chunk}\n"
            f"$R^2$={row['random_val_state_r2']:.3f}"
        )
        fig.text(pos.x0 - 0.012, (pos.y0 + pos.y1)/2,
                 row_text, ha="right", va="center", fontsize=10.2)

    fig.text(
        0.52, 0.025,
        "Examples selected by held-out $R^2$ quantiles from the 40-recording cohort; "
        "selection used only for visualization.",
        ha="center", va="center", fontsize=9.5
    )

    fig.savefig(out / "figureS_gpinn_quantile_examples_compact.png",
                dpi=300, bbox_inches="tight")
    fig.savefig(out / "figureS_gpinn_quantile_examples_compact.pdf",
                bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
