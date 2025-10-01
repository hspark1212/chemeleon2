#!/bin/bash
#SBATCH --job-name=Torch_Distributed
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=01:00:00
# SBATCH --array=0-7%1


source ~/.bashrc
mamba activate chem

module load brics/nccl
module load brics/aws-ofi-nccl

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export HYDRA_FULL_ERROR=1


export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=29500

srun -N1 --ntasks=1 --gpus=2 nvidia-smi -L

srun -N 1 --ntasks=1 \
    --gpus=2 \
    python src/train_vae.py experiment=mp_40/vae_dng.yaml paths.data_dir=/home/u5bd/lleon.u5bd/chemeleon/data trainer.devices=2 


