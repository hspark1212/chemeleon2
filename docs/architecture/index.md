# Architecture Overview

Chemeleon2 implements a three-stage generative pipeline for crystal structure generation, combining variational autoencoders, diffusion models, and reinforcement learning.

## Pipeline Overview

```{mermaid}
flowchart LR
    subgraph Stage1["Stage 1: VAE Training"]
        A[Crystal Structure] --> B[Encoder]
        B --> C[Latent Space z]
        C --> D[Decoder]
        D --> E[Reconstructed Structure]
    end

    subgraph Stage2["Stage 2: LDM Training"]
        F[Noise] --> G[DiT Denoiser]
        G --> H[Latent z]
        H --> I[VAE Decoder]
        I --> J[Generated Structure]
    end

    subgraph Stage3["Stage 3: RL Fine-tuning"]
        K[LDM] --> L[Generated Structures]
        L --> M[Reward Computation]
        M --> N[GRPO Update]
        N --> K
    end
```

## Module Responsibilities

| Module | Purpose | Key Components |
|--------|---------|----------------|
| `vae_module` | Encode/decode crystal structures | Transformer encoder, Transformer decoder |
| `ldm_module` | Diffusion-based generation | DiT denoiser, Gaussian diffusion |
| `rl_module` | Reward-guided fine-tuning | GRPO algorithm, Reward components |
| `data` | Data loading and batching | CrystalBatch, MPDataset |
| `utils` | Metrics and utilities | Metrics, Featurizer, Visualize |

## Data Flow

1. **Training Data**: Crystal structures from Materials Project (MP-20, Alex-MP-20)
2. **Encoding**: VAE converts structures to continuous latent vectors
3. **Diffusion**: LDM learns to denoise in latent space
4. **RL Optimization**: GRPO maximizes reward signals from generated structures

## Directory Structure

```
src/
├── vae_module/          # Variational Autoencoder
│   ├── vae_module.py    # Main VAE Lightning module
│   ├── encoders/        # Encoder architectures
│   └── decoders/        # Decoder architectures
├── ldm_module/          # Latent Diffusion Model
│   ├── ldm_module.py    # Main LDM Lightning module
│   ├── denoisers/       # DiT denoiser
│   └── diffusion/       # Diffusion utilities
├── rl_module/           # Reinforcement Learning
│   ├── rl_module.py     # Main RL Lightning module
│   ├── reward.py        # Reward aggregation
│   └── components.py    # Reward components
├── data/                # Data loading
│   ├── datamodule.py    # Lightning DataModule
│   └── schema.py        # CrystalBatch definition
└── utils/               # Utilities
    ├── metrics.py       # Evaluation metrics
    └── featurizer.py    # Structure featurization
```

## Learn More

- [VAE Module](vae-module.md) - Crystal structure encoding
- [LDM Module](ldm-module.md) - Diffusion-based generation
- [RL Module](rl-module.md) - Reward-guided fine-tuning
- [Data Pipeline](data-pipeline.md) - Data loading and utilities
