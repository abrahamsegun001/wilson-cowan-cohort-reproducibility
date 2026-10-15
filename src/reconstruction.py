#!/usr/bin/env python3
"""
Cohort-scale fixed-parameter gPINN reconstruction validation.

Design
------
- One independently trained cold-start gPINN per recording (40 total).
- The representative chunk is selected *before* gPINN training as the
  inference-valid chunk closest to the temporal midpoint (chunk 10), with
  lower chunk index breaking ties. This avoids reconstruction-based cherry-picking.
- Physical parameters are loaded from the already-completed frozen cohort
  regression/model-selection analysis and are NOT re-estimated here.
- Architecture/training matches the representative fixed-parameter gPINN
  validation used in the MLST methodology: random pointwise interpolation,
  60k state points, 15k held-out points, fixed physics, gradient residual,
  pseudo-derivative supervision, and L-BFGS polish.
- No sequential warm start and no continuity loss are used here. This is a
  cohort generalization test of the state representation, distinct from the
  sequential/continuity experiment.

Outputs per array task
----------------------
- one-row metrics CSV
- compact reconstruction NPZ (3 full frames, temporal mean trace, validation
  observed/predicted points)
- training history NPZ
- audit JSON describing the deterministic chunk-selection rule and fixed parameters
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--manifest", type=Path, default=Path("data/preprocessing_manifest_40.csv"))
    p.add_argument(
        "--cohort-csv",
        type=Path,
        default=Path("outputs/cohort_model_selection_40/cohort_model_selection_800.csv"),
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/processed_64_linear"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cohort_gpinn_reconstruction_40"),
    )
    p.add_argument(
        "--base-module",
        type=Path,
        default=Path(__file__).with_name("gpinn_core.py"),
    )
    p.add_argument("--target-chunk", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--l-fov", type=float, default=10.0)
    p.add_argument("--chunk-duration", type=float, default=30.0)
    p.add_argument("--n-data-pool", type=int, default=60000)
    p.add_argument("--n-val", type=int, default=15000)
    p.add_argument("--n-colloc-pool", type=int, default=60000)
    p.add_argument("--n-deriv-pool", type=int, default=20000)
    p.add_argument("--eval-deriv-points", type=int, default=12000)
    p.add_argument("--eval-pde-points", type=int, default=12000)
    p.add_argument("--eval-state-batch", type=int, default=8192)
    p.add_argument("--stageA-steps", type=int, default=2000)
    p.add_argument("--stageB-steps", type=int, default=3000)
    p.add_argument("--lbfgs-steps", type=int, default=150)
    p.add_argument("--lr-stageA", type=float, default=1e-3)
    p.add_argument("--lr-stageB", type=float, default=2e-4)
    p.add_argument("--lambda-physics", type=float, default=0.05)
    p.add_argument("--lambda-deriv", type=float, default=1e-3)
    p.add_argument("--mu-grad", type=float, default=1e-5)
    p.add_argument("--data-batch", type=int, default=4096)
    p.add_argument("--colloc-batch", type=int, default=4096)
    p.add_argument("--deriv-batch", type=int, default=1024)
    p.add_argument("--ff-dim", type=int, default=128)
    p.add_argument("--hidden-neurons", type=int, default=256)
    p.add_argument("--hidden-layers", type=int, default=6)
    p.add_argument("--scale-xy", type=float, default=25.0)
    p.add_argument("--scale-t", type=float, default=50.0)
    p.add_argument("--eps-logit", type=float, default=1e-4)
    p.add_argument("--trace-stride", type=int, default=6,
                   help="Evaluate full-field spatial mean every N frames for the figure trace.")
    return p.parse_args()


def load_base(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required base gPINN module not found: {path}")
    spec = importlib.util.spec_from_file_location("gpinn_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def choose_representative_chunk(df_rec: pd.DataFrame, target: int) -> pd.Series:
    """Deterministically choose the valid chunk nearest temporal midpoint."""
    valid = np.ones(len(df_rec), dtype=bool)
    if "criterion_valid_fraction" in df_rec:
        valid &= bool_series(df_rec["criterion_valid_fraction"]).to_numpy()
    if "criterion_conditioning" in df_rec:
        valid &= bool_series(df_rec["criterion_conditioning"]).to_numpy()

    cands = df_rec.loc[valid].copy()
    if cands.empty:
        raise RuntimeError("Recording has no chunks passing the primary inference-validity gates.")
    cands["_distance_to_target"] = (cands["chunk_idx"].astype(int) - int(target)).abs()
    cands = cands.sort_values(["_distance_to_target", "chunk_idx"], ascending=[True, True])
    return cands.iloc[0]


def predict_points(model, X, device, batch=8192):
    out = []
    model.eval()
    for start in range(0, len(X), batch):
        xb = torch.tensor(X[start:start + batch], dtype=torch.float32, device=device)
        with torch.no_grad():
            y = model.forward_E(xb).detach().cpu().numpy().ravel()
        out.append(y.astype(np.float32))
    model.train()
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


def predict_frame(base, model, ti, T, H, W, device, batch=8192):
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    tt = np.full(H * W, int(ti), dtype=np.int64)
    X = base.indices_to_coords(tt, yy.ravel(), xx.ravel(), T, H, W)
    pred = predict_points(model, X, device, batch=batch)
    return pred.reshape(H, W)


def evaluate_trace(base, model, E_scaled, device, stride, batch=8192):
    T, H, W = E_scaled.shape
    frame_ids = np.arange(0, T, max(1, int(stride)), dtype=int)
    if frame_ids[-1] != T - 1:
        frame_ids = np.r_[frame_ids, T - 1]
    pred_mean = np.empty(len(frame_ids), dtype=np.float64)
    obs_mean = E_scaled[frame_ids].mean(axis=(1, 2)).astype(np.float64)
    for j, ti in enumerate(frame_ids):
        pred_mean[j] = float(predict_frame(base, model, ti, T, H, W, device, batch).mean())
    return frame_ids, obs_mean, pred_mean


def main():
    args = parse_args()
    if not (0 <= args.task_id < 40):
        raise ValueError("task-id must be in 0..39")

    base = load_base(args.base_module)
    base.set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This production cohort gPINN job requires a CUDA GPU.")

    manifest = pd.read_csv(args.manifest).sort_values("task_id").reset_index(drop=True)
    if len(manifest) != 40:
        raise RuntimeError(f"Expected 40 manifest rows, found {len(manifest)}")
    m = manifest.loc[manifest["task_id"].astype(int) == args.task_id]
    if len(m) != 1:
        raise RuntimeError(f"Manifest task {args.task_id} not unique")
    m = m.iloc[0]
    subject, session = str(m["subject"]), str(m["session"])

    cohort = pd.read_csv(args.cohort_csv)
    rec = cohort[(cohort["subject"].astype(str) == subject) &
                 (cohort["session"].astype(str) == session)].copy()
    if len(rec) != 20:
        raise RuntimeError(f"Expected 20 cohort rows for {subject}/{session}, found {len(rec)}")
    selected = choose_representative_chunk(rec, args.target_chunk)
    chunk_idx = int(selected["chunk_idx"])

    model_kind = str(selected["selected_model"]).strip().lower()
    w_hat = float(selected["selected_w_EE"])
    D_hat = float(selected["selected_D_E"])
    h_hat = float(selected["selected_h"])

    prefix = f"sub-{subject}_ses-{session}"
    gmin, gmax, nscale = base.get_global_min_max(args.data_dir, prefix, 20)
    if nscale != 20:
        raise RuntimeError(f"Expected 20 chunks for session-global scaling; found {nscale}")
    scale = gmax - gmin + 1e-8
    fpath = base.chunk_path(args.data_dir, prefix, chunk_idx)
    if not fpath.exists():
        raise FileNotFoundError(fpath)
    raw = np.load(fpath)["dfof"].astype(np.float64)
    E_scaled = (raw - gmin) / scale
    T, H, W = E_scaled.shape

    dx_mm = args.l_fov / (W - 1)
    sigma_xy_px = base.PRIMARY_SIGMA_XY_MM / dx_mm
    E_sm = gaussian_filter(E_scaled, sigma=(2.0, sigma_xy_px, sigma_xy_px), mode="reflect")
    E0, Et, Lap, q, ylogit, valid = base.compute_pseudo_fields(
        E_sm, args.fps, args.l_fov, args.tau, args.eps_logit
    )

    # Same random seed/initialization across all 40 recordings for comparability.
    base.set_seed(args.seed)
    model = base.FixedPhysicsGPINN(
        model_kind=model_kind,
        w_EE=w_hat,
        D_E=D_hat,
        h=h_hat,
        ff_dim=args.ff_dim,
        hidden_neurons=args.hidden_neurons,
        hidden_layers=args.hidden_layers,
        scale_xy=args.scale_xy,
        scale_t=args.scale_t,
    ).to(device)

    Xd_np, Yd_np, Xv_np, Yv_np = base.random_state_split(
        E_scaled, args.n_data_pool, args.n_val, args.seed + 101 + args.target_chunk
    )
    Xc_np = base.sample_colloc_pool(
        E_scaled.shape, args.n_colloc_pool, args.seed + 202 + args.target_chunk
    )
    Xr_np, Etr_np, Lapr_np, Er_np, Xrv_np, Etrv_np, Laprv_np, Erv_np = (
        base.random_derivative_split(
            E0, Et, Lap, valid, E_scaled.shape,
            args.n_deriv_pool,
            max(args.eval_deriv_points, args.eval_pde_points),
            args.seed + 303 + args.target_chunk,
        )
    )

    Xd, Yd, Xc, Xder, Et_tgt, Lap_tgt = base.to_device_tensors(
        device, Xd_np, Yd_np, Xc_np, Xr_np, Etr_np, Lapr_np
    )
    train_args = SimpleNamespace(
        lr_stageA=args.lr_stageA,
        lr_stageB=args.lr_stageB,
        stageA_steps=args.stageA_steps,
        stageB_steps=args.stageB_steps,
        lbfgs_steps=args.lbfgs_steps,
        data_batch=args.data_batch,
        colloc_batch=args.colloc_batch,
        deriv_batch=args.deriv_batch,
        tau=args.tau,
        chunk_duration=args.chunk_duration,
        l_fov=args.l_fov,
        lambda_physics=args.lambda_physics,
        mu_grad=args.mu_grad,
        lambda_deriv=args.lambda_deriv,
    )

    print("=" * 78, flush=True)
    print(f"TASK {args.task_id:02d}: {subject} {session}", flush=True)
    print(f"Representative chunk: {chunk_idx} (target={args.target_chunk}; nearest valid rule)", flush=True)
    print(f"Fixed model: {model_kind}; w={w_hat:.6f}, D={D_hat:.6f}, h={h_hat:.6f}", flush=True)
    print(f"Device: {device}; data shape={E_scaled.shape}", flush=True)
    print("Cold-start independent gPINN; no sequential transfer; no continuity loss.", flush=True)
    print("=" * 78, flush=True)

    hist = base.train_fixed_parameter_model(
        model, Xd, Yd, Xc, Xder, Et_tgt, Lap_tgt, train_args
    )

    state_mse, state_r2, state_r = base.evaluate_state_points(
        model, Xv_np, Yv_np, device, args.eval_state_batch
    )
    nd = min(args.eval_deriv_points, len(Xrv_np))
    npde = min(args.eval_pde_points, len(Xrv_np))
    et_rmse, lap_rmse = base.evaluate_derivatives(
        model, Xrv_np[:nd], Etrv_np[:nd], Laprv_np[:nd], device, train_args
    )
    neural_pde_rmse = base.evaluate_neural_pde(
        model, Xrv_np[:npde], device, train_args
    )
    pseudo_pde = base.pseudo_pde_rmse(
        Erv_np[:npde].astype(np.float64),
        Etrv_np[:npde].astype(np.float64),
        Laprv_np[:npde].astype(np.float64),
        model_kind, w_hat, D_hat, h_hat, args.tau,
    )

    y_val_pred = predict_points(model, Xv_np, device, args.eval_state_batch)
    frame_ids = np.array([0, T // 2, T - 1], dtype=int)
    pred_frames = np.stack([
        predict_frame(base, model, int(ti), T, H, W, device, args.eval_state_batch)
        for ti in frame_ids
    ]).astype(np.float32)
    obs_frames = E_scaled[frame_ids].astype(np.float32)
    trace_ids, trace_obs, trace_pred = evaluate_trace(
        base, model, E_scaled, device, args.trace_stride, args.eval_state_batch
    )

    out = args.output_dir
    metrics_dir = out / "per_recording"
    recon_dir = out / "reconstructions"
    hist_dir = out / "histories"
    audit_dir = out / "audit"
    for d in [metrics_dir, recon_dir, hist_dir, audit_dir]:
        d.mkdir(parents=True, exist_ok=True)
    stem = f"task-{args.task_id:02d}_{subject}_{session}_chunk-{chunk_idx:02d}"

    metrics = {
        "task_id": args.task_id,
        "subject": subject,
        "session": session,
        "chunk_idx": chunk_idx,
        "target_chunk": args.target_chunk,
        "chunk_distance_from_target": abs(chunk_idx - args.target_chunk),
        "selected_model": model_kind,
        "selected_w_EE": w_hat,
        "selected_D_E": D_hat,
        "selected_h": h_hat,
        "criterion_valid_fraction": str(selected.get("criterion_valid_fraction", True)).lower() in ("true", "1", "yes"),
        "criterion_conditioning": str(selected.get("criterion_conditioning", True)).lower() in ("true", "1", "yes"),
        "random_val_state_mse": state_mse,
        "random_val_state_r2": state_r2,
        "random_val_state_r": state_r,
        "random_val_Et_rmse_vs_pseudo": et_rmse,
        "random_val_Lap_rmse_vs_pseudo": lap_rmse,
        "random_val_neural_pde_rmse": neural_pde_rmse,
        "random_val_pseudodata_pde_rmse": pseudo_pde,
        "valid_fraction_recomputed": float(valid.mean()),
        "grid_T": T,
        "grid_H": H,
        "grid_W": W,
        "sigma_xy_pixels": float(sigma_xy_px),
        "sigma_xy_mm": float(base.PRIMARY_SIGMA_XY_MM),
        "cold_start": True,
        "sequential_transfer": False,
        "continuity_loss": False,
        "seed": args.seed,
    }
    pd.DataFrame([metrics]).to_csv(metrics_dir / f"{stem}_metrics.csv", index=False)

    np.savez_compressed(
        recon_dir / f"{stem}_reconstruction.npz",
        frame_ids=frame_ids,
        frame_times_s=frame_ids.astype(np.float64) / args.fps,
        observed_frames=obs_frames,
        predicted_frames=pred_frames,
        absolute_error_frames=np.abs(pred_frames - obs_frames),
        trace_frame_ids=trace_ids,
        trace_times_s=trace_ids.astype(np.float64) / args.fps,
        trace_observed_mean=trace_obs.astype(np.float32),
        trace_predicted_mean=trace_pred.astype(np.float32),
        validation_observed=np.asarray(Yv_np).ravel().astype(np.float32),
        validation_predicted=y_val_pred.astype(np.float32),
    )
    np.savez_compressed(
        hist_dir / f"{stem}_history.npz",
        **{k: np.asarray(v, dtype=object if k == "phase" else None) for k, v in hist.items()},
    )
    audit = {
        "selection_rule": "Choose the inference-valid chunk closest to chunk 10; lower chunk index breaks ties. Selection occurs before gPINN training and is independent of reconstruction accuracy.",
        "subject": subject,
        "session": session,
        "task_id": args.task_id,
        "chosen_chunk": chunk_idx,
        "target_chunk": args.target_chunk,
        "fixed_parameters_source": str(args.cohort_csv),
        "fixed_parameters": {"model": model_kind, "w_EE": w_hat, "D_E": D_hat, "h": h_hat},
        "training": {
            "cold_start": True,
            "sequential_transfer": False,
            "continuity_loss": False,
            "stageA_steps": args.stageA_steps,
            "stageB_steps": args.stageB_steps,
            "lbfgs_steps": args.lbfgs_steps,
            "random_pointwise_validation": True,
            "interpretation": "spatiotemporal interpolation, not forecasting",
        },
    }
    (audit_dir / f"{stem}_audit.json").write_text(json.dumps(audit, indent=2))

    print("\nRESULT", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved: {stem}", flush=True)


if __name__ == "__main__":
    main()
