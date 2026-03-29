#!/usr/bin/env bash
#SBATCH --job-name=sleipner-volume
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH=src
python3 -m ccs_monitoring.cli build-volume --config configs/sleipner_volume.yaml
python3 -m ccs_monitoring.cli render-4d --config configs/sleipner_volume.yaml
