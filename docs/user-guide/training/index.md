# Training Overview

This guide covers how to train Chemeleon2 models. The framework implements a three-stage training pipeline where each stage builds upon the previous one.

## Training Pipeline

```
VAE → LDM → RL
```

| Stage | Purpose | Input | Output |
|-------|---------|-------|--------|
| **VAE** | Encode crystal structures into latent space | Crystal structures | Latent vectors |
| **LDM** | Learn to generate in latent space via diffusion | VAE checkpoint | Diffusion model |
| **RL** | Fine-tune LDM with reward functions | LDM checkpoint | Optimized generator |
| **Predictor** | Predict properties from latent vectors (optional) | VAE checkpoint | Property predictor |

## Configuration System

Chemeleon2 uses [Hydra](https://hydra.cc/) for configuration management. All configs are in the [`configs/`](https://github.com/hspark1212/chemeleon2/tree/main/configs) directory.

### Directory Structure

```
configs/
├── train_vae.yaml                              # configs/train_vae.yaml
├── train_ldm.yaml                              # configs/train_ldm.yaml
├── train_rl.yaml                               # configs/train_rl.yaml
├── train_predictor.yaml                        # configs/train_predictor.yaml
├── experiment/                                 # Experiment-specific overrides
│   ├── mp_20/
│   │   ├── vae_dng.yaml                        # configs/experiment/mp_20/vae_dng.yaml
│   │   ├── ldm_null.yaml                       # configs/experiment/mp_20/ldm_null.yaml
│   │   └── rl_dng.yaml                         # configs/experiment/mp_20/rl_dng.yaml
│   └── alex_mp_20_bandgap/
│       ├── predictor_dft_band_gap.yaml         # configs/experiment/alex_mp_20_bandgap/predictor_dft_band_gap.yaml
│       └── rl_bandgap.yaml                     # configs/experiment/alex_mp_20_bandgap/rl_bandgap.yaml
├── data/                                       # Dataset configurations
├── vae_module/                                 # VAE architecture configs
├── ldm_module/                                 # LDM architecture configs
├── rl_module/                                  # RL configs
├── trainer/                                    # PyTorch Lightning trainer
├── logger/                                     # WandB logging (configs/logger/wandb.yaml)
└── callbacks/                                  # Training callbacks
```

### Override Syntax

Override any config parameter from the command line:

```bash
# Override single parameter
python src/train_vae.py trainer.max_epochs=100

# Override multiple parameters
python src/train_ldm.py data.batch_size=64 trainer.max_epochs=500

# Use experiment config (loads all overrides from file)
python src/train_vae.py experiment=mp_20/vae_dng
```

### View Resolved Configuration

Check the fully resolved config without running training:

```bash
python src/train_ldm.py experiment=mp_20/ldm_null --cfg job
```

## Checkpoint Management

Chemeleon2 supports two ways to specify checkpoint paths.

### Hub Resolver (Recommended)

Automatically downloads pre-trained checkpoints from HuggingFace:

```yaml
# In config files
ldm_module:
    vae_ckpt_path: ${hub:mp_20_vae}
    ldm_ckpt_path: ${hub:mp_20_ldm_base}
```

```bash
# In CLI
python src/train_ldm.py ldm_module.vae_ckpt_path='${hub:mp_20_vae}'
```

### Local File Paths

Use existing checkpoint files on your system:

```yaml
# In config files
vae_ckpt_path: ckpts/mp_20/vae/my_checkpoint.ckpt
```

```bash
# In CLI
python src/train_ldm.py ldm_module.vae_ckpt_path=ckpts/my_vae.ckpt
```

### Available Pre-trained Checkpoints

| Hub Key | Description |
|---------|-------------|
| `mp_20_vae` | VAE trained on MP-20 dataset |
| `alex_mp_20_vae` | VAE trained on Alex MP-20 dataset |
| `mp_20_ldm_base` | Base LDM on MP-20 (before RL) |
| `mp_20_ldm` | LDM with RL fine-tuning on MP-20 |
| `alex_mp_20_ldm` | LDM with RL fine-tuning on Alex MP-20 |

## Experiment Tracking

Chemeleon2 uses Weights & Biases (wandb) for logging by default.

### Setup

```bash
# First time: login to wandb
wandb login
```

### Offline Mode

Run without internet connection:

```bash
WANDB_MODE=offline python src/train_vae.py experiment=mp_20/vae_dng
```

### Custom Project/Run Names

```bash
python src/train_vae.py logger.wandb.project=my_project logger.wandb.name=my_run
```

## Next Steps

- [VAE Training](vae.md) - First stage: encode crystals to latent space
- [LDM Training](ldm.md) - Second stage: diffusion model in latent space
- [RL Training](rl.md) - Third stage: fine-tune with rewards
- [Predictor Training](predictor.md) - Optional: property prediction
