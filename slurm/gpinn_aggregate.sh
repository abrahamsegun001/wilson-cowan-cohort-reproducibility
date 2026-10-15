#!/usr/bin/env bash
#SBATCH --job-name=wc_recon_agg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/wc_recon_agg_%j.out
#SBATCH --error=logs/wc_recon_agg_%j.err
set -euo pipefail
module purge
module load StdEnv/2023 python/3.10 gcc
if [[ -n "${VENV_PATH:-}" ]]; then source "$VENV_PATH/bin/activate"; fi
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
python src/reconstruction_aggregate.py \
  --input-dir outputs/cohort_gpinn_reconstruction_40 \
  --output-dir outputs/cohort_gpinn_reconstruction_40
