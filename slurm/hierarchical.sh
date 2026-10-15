#!/usr/bin/env bash
#SBATCH --job-name=wc_hierarchical
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/wc_hierarchical_%j.out
#SBATCH --error=logs/wc_hierarchical_%j.err
set -euo pipefail
module purge
module load StdEnv/2023 python/3.10 gcc
if [[ -n "${VENV_PATH:-}" ]]; then source "$VENV_PATH/bin/activate"; fi
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
python src/hierarchical.py \
  --input outputs/cohort_longitudinal_40_primary/chunk_dynamics_nearest_stable_equilibrium.csv \
  --output-dir outputs/cohort_hierarchical_40_primary
