sbatch run1.sh

# how to change embedding in 


"$@"


sbatch run1.sh vae_module.encoder.index_embedding=none \
    vae_module.decoder.index_embedding=none \
    logger.wandb.name="ab-idx none"




sbatch run1.sh vae_module.encoder.index_embedding=global_num_atoms \
    vae_module.decoder.index_embedding=global_num_atoms \
    logger.wandb.name="ab-idx glob-num"


sbatch run1.sh logger.wandb.name="ab-idx sinusoidal"


sbatch run1.sh vae_module.encoder.index_embedding=none \
    logger.wandb.name="ab-idx none-sinus"

    bash run1.sh logger.wandb.name="ab-idx none"