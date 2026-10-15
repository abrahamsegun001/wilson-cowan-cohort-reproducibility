# Changelog

## v1.0.0-rc2 — 2026-10-15

- Reorganized the public package to mirror the compact structure of the earlier single-session repository.
- Added one root Python entry point (`run_pipeline.py`) and one Slurm entry point (`submit_pipeline.sh`).
- Flattened analysis code into focused `src/` modules.
- Removed the vendored preprocessing subproject; the derivative-aware workflow remains a separate companion repository.
- Corrected the repository URL so the cohort release does not overwrite or masquerade as the earlier single-session repository.
- Retained lightweight manuscript-level derived summaries and release validation tooling.
