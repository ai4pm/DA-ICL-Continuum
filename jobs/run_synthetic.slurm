#!/bin/bash
#SBATCH --job-name=bibm-synthetic
#SBATCH --partition=campus
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=results/logs/synthetic-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p results/logs
bibm-replicate run-synthetic --config configs/synthetic_paper.yaml
bibm-replicate analyze-synthetic --result-root results/synthetic
bibm-replicate generate-figures --experiment synthetic

