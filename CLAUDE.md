# chemeleon2 Development Guidelines

Auto-generated from all feature plans. Last updated: 2025-10-03

## Active Technologies
- Python 3.11+ + Ruff (formatter + linter), pre-commit framework, GitHub Actions (001-this-repository-was)
- Python 3.11+ (already specified in pyproject.toml) + Ruff (formatter + linter), pre-commit framework, pytest, GitHub Actions (001-this-repository-was)
- Configuration files (.toml, .yaml), git repository (001-this-repository-was)

## Development Environment
- **Python**: Use `.venv/bin/python` (Python 3.11.13 in virtual environment)
- **Virtual Environment**: `.venv/` at repository root (already activated in shell)
- **Package Manager**: uv

## Project Overview
Chameleon2 is a generative ML framework for crystal structure prediction using a 3-stage pipeline:
1. **VAE Module**: Encodes crystal structures into latent space
2. **LDM Module**: Generates structures in latent space using diffusion
3. **RL Module**: Fine-tunes LDM using reinforcement learning (GRPO)

Built on PyTorch Lightning + Hydra, supports Materials Project datasets (mp-20, mp-120).

## Project Structure
```
src/
├── vae_module/     # Variational Autoencoder
├── ldm_module/     # Latent Diffusion Model (DiT-based)
├── rl_module/      # Reinforcement Learning (PPO)
├── data/           # Dataset handling & augmentation
├── utils/          # Metrics, visualization, callbacks
├── train_vae.py    # VAE training script
├── train_ldm.py    # LDM training script (requires trained VAE)
├── train_rl.py     # RL training script (requires trained LDM)
├── train_predictor.py  # Property predictor training
├── sample.py       # Sampling/generation script
└── evaluate.py     # Evaluation script

tests/
├── baseline/       # Baseline validation tests
├── unit/           # Component unit tests
├── integration/    # Integration tests
└── contract/       # Contract tests

configs/            # Hydra configuration (experiment/, data/, modules/, trainer/)
benchmarks/         # Benchmark scripts
data/               # Dataset directories (mp-20, mp-120, etc.)
.venv/              # Virtual environment - ALWAYS use this Python
```

## Key Dependencies
- PyTorch ≥2.6.0, PyTorch Lightning ≥2.5.4
- torch-geometric ≥2.6.1
- Hydra 1.3.2
- pymatgen ≥2025.6.14
- pytest ≥7.0
- wandb ≥0.21.3
- huggingface_hub ≥0.20.0 (for checkpoint downloads)

## Commands
```bash
# Training
python src/train_vae.py experiment=<vae_experiment>
python src/train_ldm.py experiment=<ldm_experiment>
python src/train_rl.py experiment=<rl_experiment>
python src/train_predictor.py experiment=<predictor_experiment>

# Inference
python src/sample.py <sampling_configs>
python src/evaluate.py <evaluation_configs>

# Testing & Linting
pytest tests/                    # Run all tests
pytest tests/ -m baseline        # Run baseline tests
pytest tests/ -m unit            # Run unit tests
ruff check .                     # Run linter
```

## Checkpoints
Model checkpoints auto-download from `hspark1212/chemeleon2-checkpoints` (HF Hub):
- First run: Downloads ~1.7GB (4 checkpoints: VAE/LDM for mp_20 & alex_mp_20)
- Cached in `./checkpoints/` with SHA256 verification
- Auto-retry on failure (3 attempts, 2s delay)

## Code Style
Python 3.11+: Follow standard conventions

## Recent Changes
- 001-this-repository-was: Added Python 3.11+ (already specified in pyproject.toml) + Ruff (formatter + linter), pre-commit framework, pytest, GitHub Actions
- 001-this-repository-was: Added Python 3.11+ + Ruff (formatter + linter), pre-commit framework, GitHub Actions

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
