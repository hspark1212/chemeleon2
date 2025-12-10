# LDM Training

The Latent Diffusion Model (LDM) is the second stage of the Chemeleon2 pipeline. It learns to generate crystal structures by denoising in the VAE's latent space.

## What LDM Does

The LDM uses a diffusion process to generate latent vectors:

```
Noise z_T → Denoise → ... → Denoise → Clean z_0 → VAE Decoder → Crystal Structure
```

Key components (see [`src/ldm_module/ldm_module.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/ldm_module/ldm_module.py)):
- **Diffusion Transformer (DiT)**: Predicts noise at each timestep
- **DDPM/DDIM Sampling**: Iteratively denoises random noise
- **Conditioning**: Optional guidance from composition or properties

## Prerequisites

LDM training requires a **trained VAE checkpoint**:

```bash
# Use hub checkpoint
ldm_module.vae_ckpt_path: ${hub:mp_20_vae}

# Or use local checkpoint
ldm_module.vae_ckpt_path: ckpts/my_vae.ckpt
```

## Quick Start

```bash
# Train unconditional LDM (src/train_ldm.py)
python src/train_ldm.py experiment=mp_20/ldm_null
```

Training script: [`src/train_ldm.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/train_ldm.py)
Example config: [`configs/experiment/mp_20/ldm_null.yaml`](https://github.com/hspark1212/chemeleon2/blob/main/configs/experiment/mp_20/ldm_null.yaml)

:::{tip}
**Example training run**: See [this W&B run](https://wandb.ai/hspark1212/chemeleon2/groups/train_ldm%2Fmp_20/runs/4tfw67aq) for a successful LDM training example on MP-20 dataset.
:::

## Training Modes

### Unconditional Generation

Generate diverse structures without any guidance:

```bash
python src/train_ldm.py experiment=mp_20/ldm_null
```

### Composition-Conditioned Generation

Guide generation with target chemical composition:

```bash
python src/train_ldm.py experiment=mp_20/ldm_composition
```

### Property-Conditioned Generation

Guide generation with target property values (e.g., band gap):

```bash
python src/train_ldm.py experiment=alex_mp_20_bandgap/ldm_bandgap
```

## Training Commands

### Basic Training

```bash
# Use experiment config
python src/train_ldm.py experiment=mp_20/ldm_null

# Override checkpoint path
python src/train_ldm.py experiment=mp_20/ldm_null \
    ldm_module.vae_ckpt_path=ckpts/my_vae.ckpt

# Override training parameters
python src/train_ldm.py experiment=mp_20/ldm_null \
    trainer.max_epochs=500 \
    data.batch_size=64
```

### LoRA Fine-tuning

Fine-tune a pre-trained LDM with Low-Rank Adaptation:

```bash
python src/train_ldm.py experiment=alex_mp_20_bandgap/ldm_bandgap_lora
```

LoRA enables efficient fine-tuning by only updating low-rank adapter weights.

## Configuration

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_diffusion_steps` | 1000 | Number of diffusion timesteps |
| `noise_schedule` | cosine | Noise schedule (linear, cosine) |
| `hidden_dim` | 512 | DiT hidden dimension |
| `num_layers` | 12 | Number of DiT layers |
| `num_heads` | 8 | Number of attention heads |

### Example Config Override

```bash
python src/train_ldm.py experiment=mp_20/ldm_null \
    ldm_module.num_diffusion_steps=500 \
    ldm_module.hidden_dim=768
```

## Available Experiments

| Experiment | Dataset | Condition | Description |
|------------|---------|-----------|-------------|
| `mp_20/ldm_null` | MP-20 | None | Unconditional generation |
| `mp_20/ldm_composition` | MP-20 | Composition | Composition-guided |
| `alex_mp_20_bandgap/ldm_bandgap` | Alex MP-20 | Band gap | Property-guided |
| `alex_mp_20_bandgap/ldm_bandgap_lora` | Alex MP-20 | Band gap | LoRA fine-tuning |

## Training Tips

### Monitoring

Key metrics to watch in WandB:
- `train/loss`: Diffusion loss (should decrease)
- `val/loss`: Validation loss (check for overfitting)

### Typical Training

- **Duration**: ~500-1000 epochs
- **Batch size**: 32-128 depending on GPU memory
- **Learning rate**: 1e-4 (default)

### Sampling During Training

The training script periodically generates samples to monitor quality. Check the generated structures in WandB.

## Next Steps

After training LDM:
1. Note the checkpoint path
2. Option A: Proceed to [RL Training](rl.md) to fine-tune with rewards
3. Option B: Use directly for generation (see [Evaluation](../evaluation.md))
