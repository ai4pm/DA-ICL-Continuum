#!/bin/bash
# Template for one prepared task/seed partition. Adapt account/QoS directives
# to the local cluster and submit one archive per authorized partition.
#SBATCH --job-name=bibm-da-icl
#SBATCH --partition=campus-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/da-icl-%j.out

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: sbatch jobs/run_npz.slurm INPUT_NPZ OUTPUT_NPZ CHECKPOINT" >&2
  exit 2
fi

cd "${SLURM_SUBMIT_DIR}"
mkdir -p results/logs "$(dirname "$2")"
bibm-replicate run-npz \
  --input "$1" \
  --output "$2" \
  --method da-icl \
  --model-path "$3" \
  --device cuda
