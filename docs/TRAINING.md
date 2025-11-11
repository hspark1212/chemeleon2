# Training Guide

This guide provides detailed instructions for training Chemeleon2 models.
The framework implements a three-stage training pipeline:

1. **VAE (Variational Autoencoder)**: Encodes crystal structures into
   latent space representations
2. **LDM (Latent Diffusion Model)**: Learns to generate in the VAE's
   latent space using diffusion
3. **RL (Reinforcement Learning)**: Fine-tunes the LDM using
   reinforcement learning with reward functions
4. **Predictor (Optional)**: Trains property predictors using the VAE
   latent vectors for property prediction

Each stage builds upon the previous one, so you'll need to train them
in order (or use pre-trained checkpoints).

## Prerequisites

### Experiment Tracking Setup

Chemeleon2 uses Weights & Biases (wandb) for logging by default.

```bash
# First time: login to wandb
wandb login
```

## 1. Train VAE (First Stage)

The VAE encodes crystal structures into a continuous latent space.

```bash
# Train VAE on MP-20 dataset
python src/train_vae.py experiment=mp_20/vae_dng

# Override training parameters
python src/train_vae.py experiment=mp_20/vae_dng trainer.max_epochs=3000 data.batch_size=128

# Resume from checkpoint
python src/train_vae.py experiment=mp_20/vae_dng ckpt_path=ckpts/vae_checkpoint.ckpt
```

**Available VAE experiments:**

- [`mp_20/vae_dng`](../configs/experiment/mp_20/vae_dng.yaml) - VAE for de novo generation on MP-20

## 2. Train LDM (Second Stage - requires trained VAE)

The LDM learns to generate structures in the VAE's latent space using diffusion models.

```bash
# Train unconditional LDM (null condition)
python src/train_ldm.py experiment=mp_20/ldm_null

# Train composition-conditioned LDM
python src/train_ldm.py experiment=mp_20/ldm_composition

# Override checkpoint path and training parameters
python src/train_ldm.py experiment=mp_20/ldm_null \
    ldm_module.vae_ckpt_path=ckpts/my_vae.ckpt \
    trainer.max_epochs=500

# Fine-tune with LoRA
python src/train_ldm.py experiment=alex_mp_20_bandgap/ldm_bandgap_lora
```

**Available LDM experiments:**

Unconditional generation:

- [`mp_20/ldm_null`](../configs/experiment/mp_20/ldm_null.yaml) - Unconditional generation on MP-20

Conditional generation:

- [`mp_20/ldm_composition`](../configs/experiment/mp_20/ldm_composition.yaml) - Conditioned on chemical composition
- [`alex_mp_20_bandgap/ldm_bandgap`](../configs/experiment/alex_mp_20_bandgap/ldm_bandgap.yaml) - Conditioned on band gap values

LoRA fine-tuning:

- [`alex_mp_20_bandgap/ldm_bandgap_lora`](../configs/experiment/alex_mp_20_bandgap/ldm_bandgap_lora.yaml) - LoRA fine-tuning for band gap

## 3. Train RL (Third Stage - requires trained LDM)

The RL module fine-tunes the LDM using reinforcement learning with reward functions.

```bash
# Fine-tune with de novo generation reward
python src/train_rl.py experiment=mp_20/rl_dng

# Fine-tune for band gap optimization
python src/train_rl.py experiment=alex_mp_20_bandgap/rl_bandgap

# Override RL hyperparameters
python src/train_rl.py experiment=mp_20/rl_dng \
    rl_module.ldm_ckpt_path=ckpts/my_ldm.ckpt \
    rl_module.rl_configs.num_group_samples=128 \
    data.batch_size=8
```

**Available RL experiments:**

- [`mp_20/rl_dng`](../configs/experiment/mp_20/rl_dng.yaml) - De novo generation reward
- [`alex_mp_20_bandgap/rl_bandgap`](../configs/experiment/alex_mp_20_bandgap/rl_bandgap.yaml) - Band gap optimization

## 4. Train Predictor (Optional)

The predictor operates in the VAE's latent space, learning to predict material properties from latent representations. This enables efficient conditional generation without retraining the generative model.

```bash
# Train band gap predictor
python src/train_predictor.py experiment=alex_mp_20_bandgap/predictor_dft_band_gap

# Override predictor parameters
python src/train_predictor.py \
    experiment=alex_mp_20_bandgap/predictor_dft_band_gap \
    predictor_module.vae.checkpoint_path=ckpts/my_vae.ckpt \
    data.batch_size=512
```

**Available Predictor experiments:**

- [`alex_mp_20_bandgap/predictor_dft_band_gap`](../configs/experiment/alex_mp_20_bandgap/predictor_dft_band_gap.yaml) - Predict DFT band gap values

## Configuration Tips

The project uses Hydra for configuration management. You can override any parameter from the command line:

### Override training parameters

```bash
python src/train_ldm.py trainer.max_epochs=100 data.batch_size=32
```

### Configuration file structure

- `configs/experiment/`: Experiment-specific configurations
- `configs/{data,vae_module,ldm_module,rl_module}/`: Component configurations
- `configs/{trainer,logger,callbacks}/`: Training infrastructure
