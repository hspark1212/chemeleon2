# LDM Module

The Latent Diffusion Model module (`src/ldm_module/`) learns to generate crystal structures by denoising in the VAE's latent space.

## Architecture

```{mermaid}
flowchart TB
    subgraph Training
        A[Latent z from VAE] --> B[Add Noise at timestep t]
        B --> C[DiT Denoiser]
        C --> D[Predict Noise]
        D --> E[Diffusion Loss]
    end

    subgraph Sampling
        F[Random Noise] --> G[Iterative Denoising]
        G --> H[Clean Latent z]
        H --> I[VAE Decoder]
        I --> J[Crystal Structure]
    end
```

## Key Classes

### LDMModule

PyTorch Lightning module for the latent diffusion model:

```python
from src.ldm_module import LDMModule

# Load pre-trained LDM
ldm = LDMModule.load_from_checkpoint("path/to/checkpoint.ckpt")

# Sample new structures
batch_gen = ldm.sample(batch, num_samples=100)
```

**Key Methods:**
- `calculate_loss(batch)` - Computes diffusion training loss
- `sample(batch)` - Generates structures via DDPM or DDIM sampling

### DiT (Diffusion Transformer)

The denoiser architecture based on Meta's DiT:

- Transformer blocks with adaptive layer norm (adaLN)
- Timestep and condition embeddings
- Support for variable-length sequences with masking

```python
# DiT configuration
denoiser:
  _target_: src.ldm_module.denoisers.DiT
  hidden_size: 512
  depth: 12
  num_heads: 8
  mlp_ratio: 4.0
```

### Gaussian Diffusion

Implements the diffusion process:

- Forward process: gradually adds noise
- Reverse process: learned denoising
- Supports DDPM and DDIM sampling

## Conditional Generation

The LDM supports conditioning on:

| Condition Type | Description | Config |
|----------------|-------------|--------|
| Composition | Chemical formula guidance | `ldm_composition` |
| Band gap | Property-conditioned generation | `ldm_bandgap` |
| Custom | Extensible condition module | Custom config |

### Classifier-Free Guidance (CFG)

```bash
# Sample with CFG
python src/sample.py \
    ldm_ckpt=path/to/ldm.ckpt \
    cfg_scale=2.0
```

## Configuration

See `configs/ldm_module/` for LDM configurations:

```yaml
# configs/ldm_module/ldm_dng.yaml
_target_: src.ldm_module.LDMModule
denoiser:
  _target_: src.ldm_module.denoisers.DiT
  hidden_size: 512
  depth: 12
diffusion:
  timesteps: 1000
  beta_schedule: "cosine"
```

## Training

```bash
# Unconditional LDM
python src/train_ldm.py experiment=mp_20/ldm_null

# Composition-conditioned LDM
python src/train_ldm.py experiment=mp_20/ldm_composition
```

See [Training Guide](../user-guide/training.md) for more details.
