#!/bin/bash
#SBATCH --job-name=hrm-maze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=/data/yuanjiayi/HRM/slurm/%x-%j.out
#SBATCH --error=/data/yuanjiayi/HRM/slurm/%x-%j.err

set -euo pipefail

cd /home/yuanjiayi/HRM
source ~/anaconda3/etc/profile.d/conda.sh
conda activate hrm

export HRM_STORAGE_ROOT=${HRM_STORAGE_ROOT:-/data/yuanjiayi/HRM}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-${HRM_STORAGE_ROOT}/wandb}

mkdir -p \
  "${HRM_STORAGE_ROOT}/checkpoints/maze-single" \
  "${HRM_STORAGE_ROOT}/outputs" \
  "${HRM_STORAGE_ROOT}/slurm" \
  "${WANDB_DIR}"

srun python pretrain.py \
  "data_path=${HRM_STORAGE_ROOT}/data/maze-30x30-hard-1k" \
  global_batch_size=${GLOBAL_BATCH_SIZE:-96} \
  epochs=${EPOCHS:-20000} \
  eval_interval=${EVAL_INTERVAL:-2000} \
  lr=1e-4 \
  puzzle_emb_lr=1e-4 \
  weight_decay=1.0 \
  puzzle_emb_weight_decay=1.0 \
  checkpoint_every_eval=True \
  "+checkpoint_path=${HRM_STORAGE_ROOT}/checkpoints/maze-single" \
  "hydra.run.dir=${HRM_STORAGE_ROOT}/outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}"
