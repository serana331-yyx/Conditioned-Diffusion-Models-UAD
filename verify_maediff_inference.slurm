#!/bin/bash
#SBATCH --job-name=maediff_infer
#SBATCH --partition=gpu3090
#SBATCH --qos=1gpu
#SBATCH --gres=gpu:gtx3090:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/work/bio/qinghanyang24/SURF/SURF-MAEDiff/maediff_infer_%j.out

set -euo pipefail
PROJECT_DIR=/gpfs/work/bio/qinghanyang24/SURF/SURF-MAEDiff
source /gpfs/work/bio/qinghanyang24/miniforge3/etc/profile.d/conda.sh
conda activate env_python
cd "$PROJECT_DIR"

CHECKPOINT="$PROJECT_DIR/outputs/maediff_smoke/fold_1/job_3069630/checkpoints/cddpm-step-000003_fold-1.ckpt"
test -f "$CHECKPOINT"
python -u verify_maediff_inference.py \
  --checkpoint "$CHECKPOINT" \
  --data /gpfs/work/bio/qinghanyang24/SURF/benchmark_data/ixi/test \
  --output "$PROJECT_DIR/outputs/inference_checks/job_${SLURM_JOB_ID}"
