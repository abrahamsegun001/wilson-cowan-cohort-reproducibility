# Derived results

`summary/` contains lightweight manuscript-level summary tables already available at release-candidate creation time.

Before making the repository public for journal submission, run:

```bash
bash scripts/collect_release_outputs.sh ~/scratch/gpinn_project
python scripts/validate_release.py
```

This adds the full 800-chunk model-selection table, the 738-chunk inference-valid primary-dynamics table, the all-800-chunk sensitivity dynamics table, longitudinal/hierarchical outputs, and the 40-recording reconstruction metrics required to reproduce the manuscript's quantitative claims.

Large neural-network checkpoints and raw imaging data are intentionally excluded.
