# Chemeleon2

A generative machine learning framework for crystals in latent space. It combines Variational Autoencoders (VAE), Latent Diffusion Models (LDM), and Reinforcement Learning (RL).

## Overview

Chemeleon2 implements a three-stage pipeline for crystal structure generation:

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

# (Optional) Install development dependencies (pytest, ruff, pyright, etc.)
uv sync --extra dev

# (Optional) Install metrics dependencies for evaluation (mace-torch, smact)
uv sync --extra metrics
```

## (Optional) Pytorch Installation with CUDA

After completing `uv sync`, install a PyTorch version compatible with your CUDA environment to prevent compatibility issues.
For version-specific installation commands, visit the [PyTorch official website](https://pytorch.org/get-started/previous-versions/).

<details>
<summary> Example </summary>
For PyTorch 2.7.0 with CUDA 12.8:
<pre><code>uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128</code></pre>

</details>

## Quick Start

### 1. Train VAE (First Stage)

The VAE encodes crystal structures into a continuous latent space.

```bash
# Train VAE on MP-20 dataset
python src/train_vae.py experiment=mp_20/vae_dng

# Override training parameters
python src/train_vae.py experiment=mp_20/vae_dng trainer.max_epochs=3000 data.batch_size=128

# Resume from checkpoint
python src/train_vae.py experiment=mp_20/vae_dng ckpt_path=ckpts/vae_checkpoint.ckpt
```

<details>
<summary>VAE experiment examples</summary>

- [`mp_20/vae_dng`](configs/experiment/mp_20/vae_dng.yaml) - VAE for de novo generation on MP-20

</details>

### 2. Train LDM (Second Stage - requires trained VAE)

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

<details>
<summary>LDM experiment examples</summary>

**Unconditional generation:**
- [`mp_20/ldm_null`](configs/experiment/mp_20/ldm_null.yaml) - Unconditional generation on MP-20

**Conditional generation:**
- [`mp_20/ldm_composition`](configs/experiment/mp_20/ldm_composition.yaml) - Conditioned on chemical composition
- [`alex_mp_20_bandgap/ldm_bandgap`](configs/experiment/alex_mp_20_bandgap/ldm_bandgap.yaml) - Conditioned on band gap values

**LoRA fine-tuning:**
- [`alex_mp_20_bandgap/ldm_bandgap_lora`](configs/experiment/alex_mp_20_bandgap/ldm_bandgap_lora.yaml) - LoRA fine-tuning for band gap

</details>

### 3. Train RL (Third Stage - requires trained LDM)

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

<details>
<summary>RL experiment examples</summary>

- [`mp_20/rl_dng`](configs/experiment/mp_20/rl_dng.yaml) - De novo generation reward
- [`alex_mp_20_bandgap/rl_bandgap`](configs/experiment/alex_mp_20_bandgap/rl_bandgap.yaml) - Band gap optimization

</details>

### 4. Train Predictor (Optional)

Train a property predictor on the VAE's latent space for conditional generation.

```bash
# Train band gap predictor
python src/train_predictor.py experiment=alex_mp_20_bandgap/predictor_dft_band_gap

# Override predictor parameters
python src/train_predictor.py experiment=alex_mp_20_bandgap/predictor_dft_band_gap \
    predictor_module.vae.checkpoint_path=ckpts/my_vae.ckpt \
    data.batch_size=512
```

### 5. Generate Samples

Generate crystal structures using a trained LDM model.

```bash
# Generate 10000 samples with 2000 batch size using DDIM sampler
python src/sample.py --num_samples=10000 --batch_size=2000
```

</details>

### 6. Evaluate Models

Evaluate generated structures against reference datasets.

```bash
# Evaluate pre-generated structures
python src/evaluate.py \
    --structure_path=outputs/dng_samples \
    --reference_dataset=mp-20 \
    --output_file=benchmark/results/my_results.csv

# Generate and evaluate in one command
python src/evaluate.py \
    --model_path=ckpts/mp_20/ldm/ldm_null.ckpt \
    --structure_path=outputs/eval_samples \
    --reference_dataset=mp-20 \
    --num_samples=10000 \
    --batch_size=2000 \
```

## Configuration

The project uses [Hydra](https://hydra.cc) for configuration management:

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

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for detailed setup instructions, development workflow, and guidelines.
