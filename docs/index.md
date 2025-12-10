---
authors:
  - name: Hyunsoo Park
    affiliation: WMD Group, Imperial College London
  - name: Aron Walsh
    affiliation: WMD Group, Imperial College London
---

# Chemeleon2

A reinforcement learning framework in latent diffusion models for crystal structure generation using group relative policy optimization (GRPO).

```{figure} ../assets/logo.png
:alt: Chemeleon2 logo
:width: 260px
:align: center
```

## Overview

Chemeleon2 implements a three-stage pipeline for crystal structure generation:

1. **VAE Module**: Encodes crystal structures into latent space representations
2. **LDM Module**: Samples crystal structures in latent space using diffusion Transformer
3. **RL Module**: Fine-tunes the LDM with reinforcement learning

```{figure} ../assets/toc.png
:alt: Chemeleon2 pipeline overview
:width: 640px
:align: center
```

## Key Features

- **Variational Autoencoder** for crystal structure encoding
- **Latent Diffusion Model** with DiT-based architecture
- **GRPO-based Reinforcement Learning** for reward optimization
- **Modular reward system** with built-in and custom components
- **Comprehensive evaluation metrics** (uniqueness, novelty, stability, diversity)

## Quick Links

- [Installation](getting-started/installation.md) - Get started with Chemeleon2
- [Training Guide](user-guide/training.md) - Train VAE, LDM, and RL models
- [Architecture](architecture/index.md) - Understand the system design
- [API Reference](api/index.md) - Detailed API documentation

## Citation

If you use Chemeleon2 in your research, please cite:

```bibtex
@article{Park2025chemeleon2,
  title={Guiding Generative Models to Uncover Diverse and Novel Crystals via Reinforcement Learning},
  author={Hyunsoo Park and Aron Walsh},
  year={2025},
  url={https://arxiv.org/abs/2511.07158}
}
```

## License

Chemeleon2 is licensed under the MIT License. See [LICENSE](https://github.com/hspark1212/chemeleon2/blob/main/LICENSE) for more information.
