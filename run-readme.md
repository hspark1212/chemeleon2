sbatch run1.sh

# how to change embedding in 


"$@"


sbatch run1.sh vae_module.encoder.index_embedding=none \
    vae_module.decoder.index_embedding=none \
    logger.wandb.name="ab-idx none"

srun --gpus=1 --time=5:15:00 --pty /bin/bash --login


sbatch run1.sh vae_module.encoder.index_embedding=global_num_atoms \
    vae_module.decoder.index_embedding=global_num_atoms \
    logger.wandb.name="ab-idx glob-num" \
    scheduler="cosine_annealing"

sbatch run1.sh vae_module.encoder.index_embedding=global_num_atoms \
    vae_module.decoder.index_embedding=global_num_atoms \
    logger.wandb.name="ab-idx glob-num - lr - mixture" \
    scheduler="mixture"

sbatch run1.sh vae_module.encoder.index_embedding=global_num_atoms \
    vae_module.decoder.index_embedding=global_num_atoms \
    logger.wandb.name="ab-idx glob-num - lr - onecycle" \
    scheduler="one_cycle"


sbatch run1.sh vae_module.encoder.index_embedding=global_num_atoms \
    vae_module.decoder.index_embedding=global_num_atoms \
    logger.wandb.name="ab-idx glob-num"


sbatch run1.sh logger.wandb.name="ab-idx sinusoidal"


sbatch run1.sh vae_module.encoder.index_embedding=none \
    logger.wandb.name="ab-idx none-sinus"

    bash run1.sh logger.wandb.name="ab-idx none"


# Latent space diffusion model

sbatch run2-ldm.sh ldm_module.vae_ckpt_path=/home/u5bd/lleon.u5bd/chemeleon2/logs/train_vae/runs/2025-09-26_21-33-21-mp40-ab-indx-glob_num_atoms/checkpoints/last.ckpt \
name="ldm-glob_num_atoms" \

sbatch run2-ldm.sh ldm_module.vae_ckpt_path=/home/u5bd/lleon.u5bd/chemeleon2/logs/train_vae/runs/2025-09-26_21-12-24-mp40-ab-indx-sinusoidal/checkpoints/epoch_149.ckpt \
name="ldm-glob_sinusoidal" 

sbatch run2-ldm.sh ldm_module.vae_ckpt_path=/home/u5bd/lleon.u5bd/chemeleon2/logs/train_vae/runs/2025-09-26_21-12-24-mp40-ab-indx-sinusoidal/checkpoints/epoch_149.ckpt \
name="ldm-glob_sinusoidal" 


sbatch run2-ldm.sh ldm_module.vae_ckpt_path=/home/u5bd/lleon.u5bd/chemeleon2/logs/train_vae/runs/2025-09-26_21-12-24-mp40-ab-indx-sinusoidal/checkpoints/epoch_149.ckpt \
name="ldm-glob_none" 




