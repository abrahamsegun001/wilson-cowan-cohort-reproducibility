#!/usr/bin/env python3
"""
Cohort-scale regression-assisted, model-selective Wilson-Cowan inference.

Purpose
-------
Apply the *frozen* one-session selector to the full BraiDyn-BC cohort:
10 subjects x 4 sessions x 20 chunks = 800 chunk-level analyses.

The selection rule is intentionally unchanged from the validated one-session
random-pointwise regression-assisted pipeline:
  - session-global min-max normalization
  - resolution-aware Gaussian pseudo-derivatives
  - unrestricted full OLS: logit(tau E_t + E) = w E + D Lap(E) + h
  - nested reduced OLS with D = 0
  - 5-fold random-pointwise nested CV
  - 1000-replicate temporal block bootstrap for D
  - half/primary/double spatial smoothing stability
  - the same conservative pre-specified gates used previously

This script does NOT train the neural-state gPINN. It addresses the cohort
reproducibility objective first, avoiding 40 expensive GPU training runs before
we know which physical model is supported in each recording.

Modes
-----
One recording (for a SLURM array task):
  python cohort_model_selection_40.py --task-id 0

Aggregate all 40 completed recording outputs:
  python cohort_model_selection_40.py --aggregate

Important interpretation
------------------------
The 800 chunks are repeated observations nested within 40 sessions and 10 mice;
they are not 800 independent biological replicates. D_E is an effective
Laplacian spatial-coupling coefficient of the reduced model, not tissue
biophysical diffusivity.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

# -----------------------------------------------------------------------------
# Frozen constants from the one-session analysis
# -----------------------------------------------------------------------------
SYNTHETIC_FOV_MM = 10.0
SYNTHETIC_GRID_N = 128
SYNTHETIC_BEST_SIGMA_PX = 2.0
PRIMARY_SIGMA_XY_MM = SYNTHETIC_BEST_SIGMA_PX * (
    SYNTHETIC_FOV_MM / (SYNTHETIC_GRID_N - 1)
)

EXPECTED_SUBJECTS = [
    "GC48", "GC51", "GC52", "GC53", "GC54",
    "GC60", "GC61", "GC63", "GC66", "GC73",
]
EXPECTED_SESSIONS = ["day0", "day1", "day7", "day15"]
EXPECTED_RECORDINGS = 40
EXPECTED_CHUNKS_PER_RECORDING = 20
EXPECTED_TOTAL_CHUNKS = EXPECTED_RECORDINGS * EXPECTED_CHUNKS_PER_RECORDING


def parse_args():
    p = argparse.ArgumentParser(
        description="Frozen regression-assisted model selection over 40 recordings"
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--task-id", type=int, default=None)
    mode.add_argument("--aggregate", action="store_true")

    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/preprocessing_manifest_40.csv"),
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed_64_linear"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cohort_model_selection_40"),
    )
    p.add_argument("--num-chunks", type=int, default=20)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--l-fov", type=float, default=10.0)
    p.add_argument("--eps-logit", type=float, default=1e-4)

    # Frozen random-pointwise nested validation and uncertainty settings.
    p.add_argument("--nested-folds", type=int, default=5)
    p.add_argument("--bootstrap-reps", type=int, default=1000)
    p.add_argument("--bootstrap-block-frames", type=int, default=30)

    # Frozen model-selection gates.
    p.add_argument("--min-valid-fraction", type=float, default=0.90)
    p.add_argument("--max-standardized-condition", type=float, default=10.0)
    p.add_argument("--min-logit-improvement-pct", type=float, default=0.10)
    p.add_argument("--min-pde-improvement-pct", type=float, default=0.10)
    p.add_argument("--min-positive-fold-fraction", type=float, default=0.60)
    p.add_argument("--max-smoothing-relative-range", type=float, default=1.00)
    p.add_argument("--w-min", type=float, default=0.5)
    p.add_argument("--w-max", type=float, default=6.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--allow-incomplete-aggregate",
        action="store_true",
        help="Write a partial aggregate instead of failing if some tasks are absent.",
    )
    return p.parse_args()


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"task_id", "subject", "session"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if len(df) != EXPECTED_RECORDINGS:
        raise ValueError(f"Expected {EXPECTED_RECORDINGS} manifest rows, found {len(df)}")
    if df["task_id"].duplicated().any():
        raise ValueError("Duplicate task_id values in manifest")
    if df[["subject", "session"]].duplicated().any():
        raise ValueError("Duplicate subject/session rows in manifest")
    if set(df["subject"].astype(str)) != set(EXPECTED_SUBJECTS):
        raise ValueError("Manifest subject set does not match the frozen 10-subject cohort")
    if set(df["session"].astype(str)) != set(EXPECTED_SESSIONS):
        raise ValueError("Manifest session set does not match day0/day1/day7/day15")
    return df.sort_values("task_id").reset_index(drop=True)


def chunk_path(data_dir: Path, prefix: str, idx: int) -> Path:
    return data_dir / f"{prefix}_chunk-{idx:02d}_dfof.npz"


def inspect_chunk_metadata(path: Path, expected_fps: float):
    with np.load(path, allow_pickle=False) as z:
        E = z["dfof"].astype(np.float32)
        fps = float(z["fps"]) if "fps" in z.files else expected_fps
        valid_mask = z["valid_mask"].astype(bool) if "valid_mask" in z.files else None
        deriv_mask = (
            z["derivative_valid_mask"].astype(bool)
            if "derivative_valid_mask" in z.files else None
        )
    if E.ndim != 3:
        raise ValueError(f"{path}: expected T x H x W data, got {E.shape}")
    if not np.isfinite(E).all():
        raise ValueError(f"{path}: non-finite dfof values")
    if not np.isclose(fps, expected_fps, atol=0.05):
        raise ValueError(f"{path}: fps={fps}, expected {expected_fps}")
    if valid_mask is not None and deriv_mask is not None:
        if valid_mask.shape != E.shape[1:] or deriv_mask.shape != E.shape[1:]:
            raise ValueError(f"{path}: mask shape mismatch")
        if np.any(deriv_mask & ~valid_mask):
            raise ValueError(f"{path}: derivative-valid mask is not a subset of valid mask")
    return E, fps, valid_mask, deriv_mask


def get_session_minmax(data_dir: Path, prefix: str, n_chunks: int, fps: float):
    gmin, gmax = float("inf"), float("-inf")
    paths = []
    shapes = []
    for i in range(n_chunks):
        p = chunk_path(data_dir, prefix, i)
        if not p.exists():
            raise FileNotFoundError(f"Missing expected chunk: {p}")
        E, _, _, _ = inspect_chunk_metadata(p, fps)
        gmin = min(gmin, float(np.min(E)))
        gmax = max(gmax, float(np.max(E)))
        paths.append(p)
        shapes.append(tuple(E.shape))
    if not np.isfinite(gmin) or not np.isfinite(gmax) or gmax <= gmin:
        raise RuntimeError(f"Invalid session normalization range: {gmin}, {gmax}")
    return gmin, gmax, paths, shapes


# -----------------------------------------------------------------------------
# Frozen pseudo-derivative / regression selector
# -----------------------------------------------------------------------------

def compute_pseudo_fields(E_sm, fps: float, l_fov: float, tau: float, eps: float):
    _, H, W = E_sm.shape
    dt = 1.0 / fps
    dx = l_fov / (W - 1)
    dy = l_fov / (H - 1)

    E0 = E_sm[1:-1, 1:-1, 1:-1].astype(np.float64, copy=False)
    Et = (
        E_sm[2:, 1:-1, 1:-1].astype(np.float64, copy=False)
        - E_sm[:-2, 1:-1, 1:-1].astype(np.float64, copy=False)
    ) / (2.0 * dt)
    d2x = (
        E_sm[1:-1, 1:-1, 2:].astype(np.float64, copy=False)
        - 2.0 * E0
        + E_sm[1:-1, 1:-1, :-2].astype(np.float64, copy=False)
    ) / (dx * dx)
    d2y = (
        E_sm[1:-1, 2:, 1:-1].astype(np.float64, copy=False)
        - 2.0 * E0
        + E_sm[1:-1, :-2, 1:-1].astype(np.float64, copy=False)
    ) / (dy * dy)
    lap = d2x + d2y
    q = tau * Et + E0
    valid = (
        np.isfinite(E0)
        & np.isfinite(Et)
        & np.isfinite(lap)
        & np.isfinite(q)
        & (q > eps)
        & (q < 1.0 - eps)
    )
    y = np.full_like(q, np.nan, dtype=np.float64)
    y[valid] = np.log(q[valid] / (1.0 - q[valid]))
    return E0, Et, lap, q, y, valid


def full_stats(E, L, y):
    E = np.asarray(E, dtype=np.float64).ravel()
    L = np.asarray(L, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = float(len(y))
    return np.array([
        np.sum(E * E), np.sum(E * L), np.sum(E),
        np.sum(L * L), np.sum(L), n,
        np.sum(E * y), np.sum(L * y), np.sum(y),
    ], dtype=np.float64)


def reduced_stats(E, y):
    E = np.asarray(E, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = float(len(y))
    return np.array([
        np.sum(E * E), np.sum(E), n, np.sum(E * y), np.sum(y)
    ], dtype=np.float64)


def solve_full_stats(s):
    XTX = np.array([
        [s[0], s[1], s[2]],
        [s[1], s[3], s[4]],
        [s[2], s[4], s[5]],
    ], dtype=np.float64)
    XTy = np.array([s[6], s[7], s[8]], dtype=np.float64)
    try:
        return np.linalg.solve(XTX, XTy)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(XTX) @ XTy


def solve_reduced_stats(s):
    XTX = np.array([[s[0], s[1]], [s[1], s[2]]], dtype=np.float64)
    XTy = np.array([s[3], s[4]], dtype=np.float64)
    try:
        return np.linalg.solve(XTX, XTy)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(XTX) @ XTy


def fit_full(E, L, y, valid):
    return solve_full_stats(full_stats(E[valid], L[valid], y[valid]))


def fit_reduced(E, y, valid):
    return solve_reduced_stats(reduced_stats(E[valid], y[valid]))


def evaluate_regression_vectors(E, Et, L, y, tau, beta_full=None, beta_reduced=None):
    E = np.asarray(E, dtype=np.float64).ravel()
    Et = np.asarray(Et, dtype=np.float64).ravel()
    L = np.asarray(L, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if beta_full is not None:
        w, D, h = map(float, beta_full)
        z = w * E + D * L + h
    else:
        w, h = map(float, beta_reduced)
        z = w * E + h
    logit_rmse = float(np.sqrt(np.mean((z - y) ** 2)))
    s = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
    pde_res = tau * Et - (-E + s)
    pde_rmse = float(np.sqrt(np.mean(pde_res ** 2)))
    return logit_rmse, pde_rmse


def regression_condition(E, L, valid):
    Ev = E[valid]
    Lv = L[valid]
    X = np.column_stack([Ev, Lv, np.ones_like(Ev)])
    raw = float(np.linalg.cond(X))
    Xstd = np.column_stack([
        (Ev - Ev.mean()) / (Ev.std() + 1e-12),
        (Lv - Lv.mean()) / (Lv.std() + 1e-12),
    ])
    std = float(np.linalg.cond(Xstd))
    corr = float(np.corrcoef(Ev, Lv)[0, 1])
    return raw, std, corr


def nested_random_cv(E, Et, L, y, valid, tau, n_folds, seed):
    Ev = E[valid].astype(np.float64, copy=False).ravel()
    Etv = Et[valid].astype(np.float64, copy=False).ravel()
    Lv = L[valid].astype(np.float64, copy=False).ravel()
    yv = y[valid].astype(np.float64, copy=False).ravel()
    n = len(yv)
    if n < max(500, n_folds * 100):
        raise RuntimeError("Too few valid points for random nested CV")
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), n_folds)
    total_full = full_stats(Ev, Lv, yv)
    total_red = reduced_stats(Ev, yv)
    rows = []
    for fold_idx, test_idx in enumerate(folds):
        test_full = full_stats(Ev[test_idx], Lv[test_idx], yv[test_idx])
        test_red = reduced_stats(Ev[test_idx], yv[test_idx])
        b_full = solve_full_stats(total_full - test_full)
        b_red = solve_reduced_stats(total_red - test_red)
        full_logit, full_pde = evaluate_regression_vectors(
            Ev[test_idx], Etv[test_idx], Lv[test_idx], yv[test_idx], tau,
            beta_full=b_full,
        )
        red_logit, red_pde = evaluate_regression_vectors(
            Ev[test_idx], Etv[test_idx], Lv[test_idx], yv[test_idx], tau,
            beta_reduced=b_red,
        )
        rows.append({
            "fold": int(fold_idx),
            "n_test": int(len(test_idx)),
            "D_train": float(b_full[1]),
            "w_full_train": float(b_full[0]),
            "h_full_train": float(b_full[2]),
            "w_reduced_train": float(b_red[0]),
            "h_reduced_train": float(b_red[1]),
            "full_logit_rmse": full_logit,
            "reduced_logit_rmse": red_logit,
            "logit_improvement_pct": float(
                100.0 * (red_logit - full_logit) / (red_logit + 1e-12)
            ),
            "full_pde_rmse": full_pde,
            "reduced_pde_rmse": red_pde,
            "pde_improvement_pct": float(
                100.0 * (red_pde - full_pde) / (red_pde + 1e-12)
            ),
        })
    return rows


def temporal_block_bootstrap_D(E, L, y, valid, block_frames, reps, seed):
    blocks = []
    n_frames = E.shape[0]
    for start in range(0, n_frames, block_frames):
        end = min(n_frames, start + block_frames)
        vb = valid[start:end]
        if int(vb.sum()) < 100:
            continue
        blocks.append(full_stats(E[start:end][vb], L[start:end][vb], y[start:end][vb]))
    if len(blocks) < 5:
        raise RuntimeError("Too few temporal blocks for D_E bootstrap")
    block_stats = np.stack(blocks, axis=0)
    rng = np.random.default_rng(seed)
    B = len(blocks)
    vals = np.empty(reps, dtype=np.float64)
    for r in range(reps):
        draw = rng.integers(0, B, size=B)
        vals[r] = solve_full_stats(block_stats[draw].sum(axis=0))[1]
    return {
        "n_blocks": int(B),
        "mean": float(vals.mean()),
        "sd": float(vals.std(ddof=1)),
        "ci95_low": float(np.quantile(vals, 0.025)),
        "ci95_high": float(np.quantile(vals, 0.975)),
        "fraction_positive": float(np.mean(vals > 0.0)),
    }


def smoothing_D_estimates(E_scaled, fps, l_fov, tau, eps):
    _, _, W = E_scaled.shape
    dx = l_fov / (W - 1)
    primary_px = PRIMARY_SIGMA_XY_MM / dx
    results = {}
    for label, factor in [("half", 0.5), ("primary", 1.0), ("double", 2.0)]:
        sigma_px = factor * primary_px
        sm = gaussian_filter(E_scaled, sigma=(2.0, sigma_px, sigma_px), mode="reflect")
        E0, Et, L, _, y, valid = compute_pseudo_fields(sm, fps, l_fov, tau, eps)
        if int(valid.sum()) < 100:
            raise RuntimeError(f"Too few valid points at smoothing={label}")
        beta = fit_full(E0, L, y, valid)
        results[label] = {
            "D": float(beta[1]), "w": float(beta[0]), "h": float(beta[2]),
            "sigma_px": float(sigma_px),
            "sigma_mm": float(factor * PRIMARY_SIGMA_XY_MM),
        }
    Ds = np.array([results[k]["D"] for k in ("half", "primary", "double")])
    primary_D = results["primary"]["D"]
    rel_range = float((Ds.max() - Ds.min()) / (abs(primary_D) + 1e-8))
    all_positive = bool(np.all(Ds > 0.0))
    return results, rel_range, all_positive


def select_one_chunk(E_scaled, args, task_id: int, chunk_idx: int):
    T, H, W = E_scaled.shape
    dx_mm = args.l_fov / (W - 1)
    sigma_xy_px = PRIMARY_SIGMA_XY_MM / dx_mm
    E_sm = gaussian_filter(
        E_scaled,
        sigma=(2.0, sigma_xy_px, sigma_xy_px),
        mode="reflect",
    )
    E0, Et, Lap, _, ylogit, valid = compute_pseudo_fields(
        E_sm, args.fps, args.l_fov, args.tau, args.eps_logit
    )
    if int(valid.sum()) < 500:
        raise RuntimeError(f"Task {task_id} chunk {chunk_idx}: too few valid points")

    valid_fraction = float(valid.mean())
    raw_cond, std_cond, corr = regression_condition(E0, Lap, valid)
    b_full = fit_full(E0, Lap, ylogit, valid)
    b_red = fit_reduced(E0, ylogit, valid)

    # Task 0 reproduces the original GC48/day0 random seeds exactly.
    task_seed_offset = int(task_id) * 100000
    cv_rows = nested_random_cv(
        E0, Et, Lap, ylogit, valid, args.tau, args.nested_folds,
        args.seed + task_seed_offset + 5000 + chunk_idx,
    )
    logit_imps = np.array([r["logit_improvement_pct"] for r in cv_rows])
    pde_imps = np.array([r["pde_improvement_pct"] for r in cv_rows])
    boot = temporal_block_bootstrap_D(
        E0, Lap, ylogit, valid,
        args.bootstrap_block_frames, args.bootstrap_reps,
        args.seed + task_seed_offset + 10000 + chunk_idx,
    )
    smooth, smooth_rel_range, smooth_all_positive = smoothing_D_estimates(
        E_scaled, args.fps, args.l_fov, args.tau, args.eps_logit
    )

    criteria = {
        "valid_fraction": valid_fraction >= args.min_valid_fraction,
        "conditioning": std_cond <= args.max_standardized_condition,
        "D_point_positive": bool(float(b_full[1]) > 0.0),
        "D_CI95_positive": bool(boot["ci95_low"] > 0.0),
        "logit_mean_improvement": float(logit_imps.mean()) >= args.min_logit_improvement_pct,
        "pde_mean_improvement": float(pde_imps.mean()) >= args.min_pde_improvement_pct,
        "logit_fold_consistency": float(np.mean(logit_imps > 0.0)) >= args.min_positive_fold_fraction,
        "pde_fold_consistency": float(np.mean(pde_imps > 0.0)) >= args.min_positive_fold_fraction,
        "smoothing_all_positive": smooth_all_positive,
        "smoothing_relative_range": smooth_rel_range <= args.max_smoothing_relative_range,
        "w_in_bounds": args.w_min <= float(b_full[0]) <= args.w_max,
    }
    use_full = bool(all(criteria.values()))
    selected_model = "full" if use_full else "reduced"
    if use_full:
        w_hat, D_hat, h_hat = map(float, b_full)
    else:
        w_hat, h_hat = map(float, b_red)
        D_hat = 0.0

    row = {
        "chunk_idx": int(chunk_idx),
        "time_start_min": float(0.5 * chunk_idx),
        "time_end_min": float(0.5 * (chunk_idx + 1)),
        "selected_model": selected_model,
        "selected_w_EE": w_hat,
        "selected_D_E": D_hat,
        "selected_h": h_hat,
        "full_regression_w_EE": float(b_full[0]),
        "full_regression_D_E": float(b_full[1]),
        "full_regression_h": float(b_full[2]),
        "reduced_regression_w_EE": float(b_red[0]),
        "reduced_regression_h": float(b_red[1]),
        "D_boot_mean": boot["mean"],
        "D_boot_sd": boot["sd"],
        "D_boot_ci95_low": boot["ci95_low"],
        "D_boot_ci95_high": boot["ci95_high"],
        "D_boot_fraction_positive": boot["fraction_positive"],
        "nested_logit_improvement_pct": float(logit_imps.mean()),
        "nested_pde_improvement_pct": float(pde_imps.mean()),
        "nested_logit_positive_fold_fraction": float(np.mean(logit_imps > 0.0)),
        "nested_pde_positive_fold_fraction": float(np.mean(pde_imps > 0.0)),
        "valid_fraction": valid_fraction,
        "condition_raw": raw_cond,
        "condition_standardized": std_cond,
        "corr_E_laplacian": corr,
        "smoothing_D_half": smooth["half"]["D"],
        "smoothing_D_primary": smooth["primary"]["D"],
        "smoothing_D_double": smooth["double"]["D"],
        "smoothing_D_relative_range": smooth_rel_range,
        "state_mean_scaled": float(np.mean(E_scaled)),
        "state_sd_scaled": float(np.std(E_scaled)),
        "grid_T": int(T), "grid_H": int(H), "grid_W": int(W),
        "dx_mm": float(dx_mm),
        "sigma_xy_pixels": float(sigma_xy_px),
        "sigma_xy_mm": float(PRIMARY_SIGMA_XY_MM),
    }
    row.update({f"criterion_{k}": bool(v) for k, v in criteria.items()})
    detail = {
        "chunk_idx": int(chunk_idx),
        "selected_model": selected_model,
        "criteria": criteria,
        "full_beta": {"w_EE": float(b_full[0]), "D_E": float(b_full[1]), "h": float(b_full[2])},
        "reduced_beta": {"w_EE": float(b_red[0]), "D_E": 0.0, "h": float(b_red[1])},
        "bootstrap_D": boot,
        "nested_random_cv": cv_rows,
        "smoothing": smooth,
        "smoothing_relative_range": smooth_rel_range,
        "valid_fraction": valid_fraction,
        "condition_raw": raw_cond,
        "condition_standardized": std_cond,
        "corr_E_laplacian": corr,
    }
    return row, detail


def recording_stem(task_id: int, subject: str, session: str) -> str:
    return f"task-{task_id:02d}_{subject}_{session}"


def run_task(args):
    manifest = load_manifest(args.manifest)
    if args.task_id < 0 or args.task_id >= len(manifest):
        raise ValueError(f"--task-id must be 0..{len(manifest)-1}")
    rec = manifest.iloc[int(args.task_id)]
    task_id = int(rec["task_id"])
    subject = str(rec["subject"])
    session = str(rec["session"])
    prefix = f"sub-{subject}_ses-{session}"

    if args.num_chunks != EXPECTED_CHUNKS_PER_RECORDING:
        raise ValueError(
            f"Frozen cohort analysis expects {EXPECTED_CHUNKS_PER_RECORDING} chunks; "
            f"got {args.num_chunks}"
        )

    stem = recording_stem(task_id, subject, session)
    per_dir = args.output_dir / "per_recording"
    per_dir.mkdir(parents=True, exist_ok=True)

    gmin, gmax, paths, shapes = get_session_minmax(
        args.data_dir, prefix, args.num_chunks, args.fps
    )
    if len(set(shapes)) != 1:
        raise RuntimeError(f"{stem}: inconsistent chunk shapes: {sorted(set(shapes))}")

    rows, details = [], []
    for chunk_idx, p in enumerate(paths):
        print(f"[{stem}] chunk {chunk_idx:02d}/{args.num_chunks-1:02d}: {p.name}", flush=True)
        E, fps, valid_mask, deriv_mask = inspect_chunk_metadata(p, args.fps)
        E_scaled = ((E - gmin) / (gmax - gmin + 1e-8)).astype(np.float32)
        row, detail = select_one_chunk(E_scaled, args, task_id, chunk_idx)
        row.update({
            "task_id": task_id,
            "subject": subject,
            "session": session,
            "subject_session": f"{subject}_{session}",
            "fps": float(fps),
            "session_global_min": float(gmin),
            "session_global_max": float(gmax),
            "geometry_valid_fraction": float(valid_mask.mean()) if valid_mask is not None else np.nan,
            "derivative_valid_fraction_mask": float(deriv_mask.mean()) if deriv_mask is not None else np.nan,
        })
        details.append(detail)
        rows.append(row)
        print(
            f"  selected={row['selected_model']} "
            f"w={row['selected_w_EE']:.5f} D={row['selected_D_E']:+.5f} "
            f"h={row['selected_h']:.5f} "
            f"D95=[{row['D_boot_ci95_low']:+.5f},{row['D_boot_ci95_high']:+.5f}]",
            flush=True,
        )

    df = pd.DataFrame(rows).sort_values("chunk_idx")
    if len(df) != EXPECTED_CHUNKS_PER_RECORDING:
        raise RuntimeError(f"{stem}: expected 20 rows, got {len(df)}")
    csv_path = per_dir / f"{stem}_model_selection.csv"
    df.to_csv(csv_path, index=False)

    n_full = int((df["selected_model"] == "full").sum())
    summary = {
        "task_id": task_id,
        "subject": subject,
        "session": session,
        "prefix": prefix,
        "n_chunks": int(len(df)),
        "n_full_selected": n_full,
        "n_reduced_selected": int(len(df) - n_full),
        "fraction_full_selected": float(n_full / len(df)),
        "mean_selected_w_EE": float(df["selected_w_EE"].mean()),
        "sd_selected_w_EE": float(df["selected_w_EE"].std(ddof=1)),
        "mean_selected_D_E": float(df["selected_D_E"].mean()),
        "mean_selected_h": float(df["selected_h"].mean()),
        "n_full_D_ci_positive": int((df["D_boot_ci95_low"] > 0).sum()),
        "session_global_min": float(gmin),
        "session_global_max": float(gmax),
        "frozen_selector": {
            "nested_folds": args.nested_folds,
            "bootstrap_reps": args.bootstrap_reps,
            "bootstrap_block_frames": args.bootstrap_block_frames,
            "min_valid_fraction": args.min_valid_fraction,
            "max_standardized_condition": args.max_standardized_condition,
            "min_logit_improvement_pct": args.min_logit_improvement_pct,
            "min_pde_improvement_pct": args.min_pde_improvement_pct,
            "min_positive_fold_fraction": args.min_positive_fold_fraction,
            "max_smoothing_relative_range": args.max_smoothing_relative_range,
            "w_bounds": [args.w_min, args.w_max],
        },
        "interpretation": (
            "Descriptive within-recording summary. Chunks are repeated observations, "
            "not independent biological replicates."
        ),
    }
    (per_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
    (per_dir / f"{stem}_details.json").write_text(json.dumps(details, indent=2))
    print(f"\nSaved {csv_path}")


def aggregate(args):
    manifest = load_manifest(args.manifest)
    per_dir = args.output_dir / "per_recording"
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    frames, missing = [], []
    for _, rec in manifest.iterrows():
        task_id = int(rec["task_id"])
        subject = str(rec["subject"])
        session = str(rec["session"])
        stem = recording_stem(task_id, subject, session)
        p = per_dir / f"{stem}_model_selection.csv"
        if not p.exists():
            missing.append(str(p))
            continue
        d = pd.read_csv(p)
        if len(d) != EXPECTED_CHUNKS_PER_RECORDING:
            raise RuntimeError(f"{p}: expected 20 rows, found {len(d)}")
        frames.append(d)

    if missing and not args.allow_incomplete_aggregate:
        raise RuntimeError(
            f"Missing {len(missing)} recording outputs. First missing: {missing[0]}"
        )
    if not frames:
        raise RuntimeError("No per-recording result CSVs found")

    df = pd.concat(frames, ignore_index=True)
    key = ["task_id", "subject", "session", "chunk_idx"]
    if df[key].duplicated().any():
        raise RuntimeError("Duplicate task/subject/session/chunk rows in aggregate")
    df = df.sort_values(key).reset_index(drop=True)

    expected_rows = EXPECTED_TOTAL_CHUNKS if not missing else len(frames) * 20
    if len(df) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} aggregate rows, found {len(df)}")

    master_path = out / "cohort_model_selection_800.csv"
    df.to_csv(master_path, index=False)

    def summarize_by(group_cols):
        rows = []
        for keys, g in df.groupby(group_cols, sort=True, observed=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_cols, keys))
            row.update({
                "n_chunks": int(len(g)),
                "n_full_selected": int((g["selected_model"] == "full").sum()),
                "fraction_full_selected": float((g["selected_model"] == "full").mean()),
                "mean_selected_w_EE": float(g["selected_w_EE"].mean()),
                "sd_selected_w_EE": float(g["selected_w_EE"].std(ddof=1)),
                "mean_selected_D_E": float(g["selected_D_E"].mean()),
                "mean_full_regression_D_E": float(g["full_regression_D_E"].mean()),
                "sd_full_regression_D_E": float(g["full_regression_D_E"].std(ddof=1)),
                "mean_selected_h": float(g["selected_h"].mean()),
                "sd_selected_h": float(g["selected_h"].std(ddof=1)),
                "fraction_D_ci95_positive": float((g["D_boot_ci95_low"] > 0).mean()),
                "mean_logit_improvement_pct": float(g["nested_logit_improvement_pct"].mean()),
                "mean_pde_improvement_pct": float(g["nested_pde_improvement_pct"].mean()),
                "max_condition_standardized": float(g["condition_standardized"].max()),
            })
            rows.append(row)
        return pd.DataFrame(rows)

    recording_summary = summarize_by(["task_id", "subject", "session"])
    recording_summary.to_csv(out / "recording_summary_40.csv", index=False)

    subject_summary = summarize_by(["subject"])
    subject_summary.to_csv(out / "subject_summary_10.csv", index=False)

    session_summary = summarize_by(["session"])
    session_summary["session"] = pd.Categorical(
        session_summary["session"], categories=EXPECTED_SESSIONS, ordered=True
    )
    session_summary = session_summary.sort_values("session")
    session_summary.to_csv(out / "longitudinal_session_summary_4.csv", index=False)

    n_full = int((df["selected_model"] == "full").sum())
    summary = {
        "n_recordings_completed": int(df[["task_id", "subject", "session"]].drop_duplicates().shape[0]),
        "n_subjects": int(df["subject"].nunique()),
        "n_sessions_labels": int(df["session"].nunique()),
        "n_chunk_rows": int(len(df)),
        "n_full_selected": n_full,
        "n_reduced_selected": int(len(df) - n_full),
        "fraction_full_selected_descriptive": float(n_full / len(df)),
        "n_chunks_D_boot_ci95_positive": int((df["D_boot_ci95_low"] > 0).sum()),
        "mean_full_regression_D_E": float(df["full_regression_D_E"].mean()),
        "sd_full_regression_D_E": float(df["full_regression_D_E"].std(ddof=1)),
        "mean_selected_w_EE": float(df["selected_w_EE"].mean()),
        "mean_selected_h": float(df["selected_h"].mean()),
        "missing_recording_outputs": missing,
        "hierarchy": "chunk nested within recording/session nested within mouse",
        "statistical_guardrail": (
            "Do not treat the 800 chunk rows as independent biological replicates. "
            "Inferential longitudinal analysis should account for repeated measures within mouse."
        ),
        "selector_status": "frozen from one-session analysis; no cohort retuning",
    }
    (out / "cohort_model_selection_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nMaster table: {master_path}")


def main():
    args = parse_args()
    if args.aggregate:
        aggregate(args)
    else:
        run_task(args)


if __name__ == "__main__":
    main()
