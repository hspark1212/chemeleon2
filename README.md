# Chameleon2

A generative machine learning framework for crystal structure prediction using Variational Autoencoders (VAE), Latent Diffusion Models (LDM), and Reinforcement Learning (RL).

## Overview

Chameleon2 implements a three-stage pipeline for crystal structure generation:

1. **VAE Module**: Encodes crystal structures into latent space representations
2. **LDM Module**: Learns to generate in the VAE's latent space using diffusion
3. **RL Module**: Fine-tunes the LDM using reinforcement learning with reward functions

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd chemeleon2

# Install dependencies with uv
uv sync
```

## (Optional) Pytorch Installation with CUDA

After completing `uv sync`, install a PyTorch version compatible with your CUDA environment to prevent compatibility issues.
For version-specific installation commands, visit the [PyTorch official website](https://pytorch.org/get-started/previous-versions/).

<details>
<summary> Example </summary>
For PyTorch 2.7.0 with CUDA 12.8:
<pre><code>pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128</code></pre>

</details>

## Quick Start

### Train VAE

```bash
python src/train_vae.py experiment=<vae_experiment>
```

### Train LDM (requires trained VAE)

```bash
python src/train_ldm.py experiment=<ldm_experiment>
```

### Train RL (requires trained LDM)

```bash
python src/train_rl.py experiment=<rl_experiment>
```

### Train Predictor (optional, for property prediction)

```bash
python src/train_predictor.py experiment=<predictor_experiment>
```

### Generate Samples

```bash
python src/sample.py <sampling_configs>
```

### Evaluate Models

```bash
python src/evaluate.py <evaluation_configs>
```

## Configuration

The project uses Hydra for configuration management:

- `configs/experiment/`: Experiment-specific configurations
- `configs/{data,vae_module,ldm_module,rl_module}/`: Component configurations
- `configs/{trainer,logger,callbacks}/`: Training infrastructure

### Example: Override parameters

```bash
python src/train_ldm.py trainer.max_epochs=100 data.batch_size=32
```

### Example: Resume from checkpoint

```bash
python src/train_ldm.py ckpt_path=/path/to/checkpoint.ckpt
```

## Architecture

- Built on PyTorch Lightning and Hydra
- Modular design with three training pipelines
- Supports Materials Project datasets (mp-20, mp-120, etc.)
- Uses DiT (Diffusion Transformer) for denoising
- Implements PPO for reinforcement learning

## Key Dependencies

- PyTorch Lightning ≥2.1.0
- Hydra ≥1.3.2
- PyTorch ≥2.1.0
- torch-geometric
- pymatgen
- wandb
