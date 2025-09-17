#!/bin/bash
#SBATCH --job-name=Torch_Distributed
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=01:00:00
# SBATCH --array=0-7%1


source ~/.bashrc
mamba activate chem

# module load brics/nccl
# module load brics/aws-ofi-nccl

module load brics/nccl
module load brics/aws-ofi-nccl

module load cuda/12.6


# export MASTER_ADDR=$(hostname -s)
# export MASTER_PORT=29500

HYDRA_FULL_ERROR=1
NCCL_DEBUG=INFO

# srun -N1 --ntasks=1 --gpus=2 nvidia-smi -L

export TMPDIR=/tmp/$USER
mkdir -p $TMPDIR


mkdir gpu_logs

LOGFILE="gpu_logs/gpu_usage_${SLURM_JOB_ID}.log"


{ while true; do
    echo "========== $(date) ==========" >> "$LOGFILE"
    nvidia-smi >> "$LOGFILE"
    echo "" >> "$LOGFILE"
    sleep 20   # log every 30 seconds
  done; } &
echo "Logger PID: $!"

  
# srun -N 1 \
#     --gpus=2 \
#     --ntasks-per-node=1 \
#     --gpus-per-node=2 \
  srun  python -u src/train_vae.py \
        experiment=mp_40/vae_dng.yaml \
        paths.data_dir=/home/u5bd/lleon.u5bd/chemeleon/data \
    trainer.devices=-1\


# srun -N1 -n2 --gpus-per-task=1 \
#   python -m src.train_vae \
#     trainer.accelerator=gpu trainer.strategy=ddp trainer.devices=1