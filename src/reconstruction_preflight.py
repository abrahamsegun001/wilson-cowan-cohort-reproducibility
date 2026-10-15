#!/usr/bin/env python3
"""Fast CPU-side checks before submitting the 40-GPU array."""
from pathlib import Path
import pandas as pd
import numpy as np

MANIFEST = Path("data/preprocessing_manifest_40.csv")
COHORT = Path("outputs/cohort_model_selection_40/cohort_model_selection_800.csv")
DATA = Path("data/processed_64_linear")
BASE = Path(__file__).with_name("gpinn_core.py")
TARGET = 10


def bseries(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])

for p in [MANIFEST, COHORT, BASE]:
    if not p.exists():
        raise SystemExit(f"MISSING REQUIRED FILE: {p}")

m = pd.read_csv(MANIFEST).sort_values("task_id")
c = pd.read_csv(COHORT)
if len(m) != 40:
    raise SystemExit(f"Expected 40 manifest rows, got {len(m)}")
if len(c) != 800:
    raise SystemExit(f"Expected 800 cohort rows, got {len(c)}")

rows = []
problems = []
for _, r in m.iterrows():
    subject, session = str(r.subject), str(r.session)
    x = c[(c.subject.astype(str) == subject) & (c.session.astype(str) == session)].copy()
    if len(x) != 20:
        problems.append(f"{subject} {session}: cohort rows={len(x)}")
        continue
    valid = np.ones(len(x), dtype=bool)
    if "criterion_valid_fraction" in x:
        valid &= bseries(x.criterion_valid_fraction).to_numpy()
    if "criterion_conditioning" in x:
        valid &= bseries(x.criterion_conditioning).to_numpy()
    y = x.loc[valid].copy()
    if y.empty:
        problems.append(f"{subject} {session}: no inference-valid chunk")
        continue
    y["dist"] = (y.chunk_idx.astype(int) - TARGET).abs()
    z = y.sort_values(["dist", "chunk_idx"]).iloc[0]
    ch = int(z.chunk_idx)
    prefix = f"sub-{subject}_ses-{session}"
    needed = [DATA / f"{prefix}_chunk-{i:02d}_dfof.npz" for i in range(20)]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        problems.append(f"{subject} {session}: missing {len(missing)}/20 processed chunks; first={missing[0]}")
    rows.append({
        "task_id": int(r.task_id), "subject": subject, "session": session,
        "chosen_chunk": ch, "distance_from_chunk10": abs(ch-TARGET),
        "selected_model": z.selected_model,
        "selected_w_EE": z.selected_w_EE,
        "selected_D_E": z.selected_D_E,
        "selected_h": z.selected_h,
    })

if problems:
    print("PREFLIGHT FAILED")
    for q in problems:
        print(" -", q)
    raise SystemExit(2)

out = Path("outputs/cohort_gpinn_reconstruction_40")
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out / "representative_chunk_selection_40_PRETRAINING.csv", index=False)
print("PREFLIGHT PASSED")
print("40 recordings have a deterministic inference-valid representative chunk.")
print("Selection written to:", out / "representative_chunk_selection_40_PRETRAINING.csv")
print(pd.DataFrame(rows)[["subject","session","chosen_chunk","distance_from_chunk10"]].to_string(index=False))
