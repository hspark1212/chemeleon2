# train vae with default settings
python src/train_vae.py

# experiments with different hyperparameters
python src/train_vae.py trainer.gradient_clip_val=1.0 trainer.max_epochs=1000
python src/train_vae.py vae_module.loss_weights.kl=1e-4 logger.wandb.name=kl_1e-4 trainer.max_epochs=1000
python src/train_vae.py vae_module.optimizer.weight_decay=1e-2 logger.wandb.name=wd_1e-2 trainer.max_epochs=1000
python src/train_vae.py vae_module.noise.corruption_scale=0.5 logger.wandb.name=noise_corruption_scale_0.5 trainer.max_epochs=1000
python src/train_vae.py vae_module.noise.ratio=0 logger.wandb.name=noise_ratio_0 trainer.max_epochs=1000
python src/train_vae.py vae_module.encoder.dropout=0.1 vae_module.decoder.dropout=0.1 logger.wandb.name=transformer_dropout_0.1 trainer.max_epochs=1000