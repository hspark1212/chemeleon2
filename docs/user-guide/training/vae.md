# VAE Training

The Variational Autoencoder (VAE) is the first stage of the Chemeleon2 pipeline. It learns to encode crystal structures into a continuous latent space.

## What VAE Does

The VAE compresses crystal structure information into fixed-size latent vectors:

```
Crystal Structure (atoms, coords, lattice) → Encoder → Latent Vector z → Decoder → Reconstructed Structure
```

Key components (see [`src/vae_module/vae_module.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/vae_module/vae_module.py)):
- **Encoder**: Graph neural network that processes atomic positions and lattice
- **Latent Space**: Continuous representation where diffusion operates
- **Decoder**: Reconstructs crystal structures from latent vectors

## Quick Start

```bash
# Train VAE on MP-20 dataset (src/train_vae.py)
python src/train_vae.py experiment=mp_20/vae_dng
```

Training script: [`src/train_vae.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/train_vae.py)
Example config: [`configs/experiment/mp_20/vae_dng.yaml`](https://github.com/hspark1212/chemeleon2/blob/main/configs/experiment/mp_20/vae_dng.yaml)

:::{tip}
**Example training run**: See [this W&B run](https://wandb.ai/hspark1212/chemeleon2/groups/train_vae%2Fmp_20/runs/m4owq4i5) for a successful VAE training example on MP-20 dataset.
:::

## Training Commands

### Basic Training

```bash
# Use experiment config
python src/train_vae.py experiment=mp_20/vae_dng

# Override training parameters
python src/train_vae.py experiment=mp_20/vae_dng \
    trainer.max_epochs=3000 \
    data.batch_size=128
```

### Resume from Checkpoint

```bash
python src/train_vae.py experiment=mp_20/vae_dng \
    ckpt_path=ckpts/vae_checkpoint.ckpt
```

## Configuration

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_dim` | 256 | Dimension of latent space |
| `hidden_dim` | 256 | Hidden dimension in encoder/decoder |
| `num_layers` | 4 | Number of message passing layers |
| `kl_weight` | 1e-4 | KL divergence loss weight |

### Example Config Override

```bash
python src/train_vae.py experiment=mp_20/vae_dng \
    vae_module.latent_dim=512 \
    vae_module.kl_weight=1e-3
```

## Available Experiments

| Experiment | Dataset | Description |
|------------|---------|-------------|
| `mp_20/vae_dng` | MP-20 | VAE for de novo generation |

## Training Tips

### Monitoring

Key metrics to watch in WandB:
- `train/recon_loss`: Reconstruction loss (should decrease)
- `train/kl_loss`: KL divergence (should stabilize)
- `val/recon_loss`: Validation reconstruction (check for overfitting)

### Typical Training

- **Duration**: ~1000-3000 epochs
- **Batch size**: 64-256 depending on GPU memory
- **Learning rate**: 1e-4 (default)

### Checkpoint Naming

After training, checkpoints are saved in:
```
ckpts/{experiment_name}/vae/
```

## Next Steps

After training VAE:
1. Note the checkpoint path
2. Proceed to [LDM Training](ldm.md) using your VAE checkpoint
