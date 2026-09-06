#!/bin/bash
#SBATCH --job-name=hrm-arc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
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

ARC_DATA_PATH=${ARC_DATA_PATH:-${HRM_STORAGE_ROOT}/data/arc-aug-1000}

mkdir -p \
  "${HRM_STORAGE_ROOT}/checkpoints/arc-single" \
  "${HRM_STORAGE_ROOT}/outputs" \
  "${HRM_STORAGE_ROOT}/slurm" \
  "${WANDB_DIR}"

if [[ ! -f "${ARC_DATA_PATH}/train/dataset.json" ]]; then
  python dataset/build_arc_dataset.py --output-dir "${ARC_DATA_PATH}"
fi

srun python pretrain.py \
  "data_path=${ARC_DATA_PATH}" \
  global_batch_size=${GLOBAL_BATCH_SIZE:-96} \
  epochs=${EPOCHS:-12500} \
  eval_interval=${EVAL_INTERVAL:-5000} \
  checkpoint_every_eval=True \
  "+checkpoint_path=${HRM_STORAGE_ROOT}/checkpoints/arc-single" \
  "hydra.run.dir=${HRM_STORAGE_ROOT}/outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}"
