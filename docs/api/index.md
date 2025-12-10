# API Reference

This section provides API documentation for all Chemeleon2 modules.

## Module Index

### Core Modules

| Module | Description |
|--------|-------------|
| `src.vae_module` | Variational Autoencoder for crystal structure encoding |
| `src.ldm_module` | Latent Diffusion Model for generation |
| `src.rl_module` | Reinforcement Learning fine-tuning |

### Data & Utilities

| Module | Description |
|--------|-------------|
| `src.data` | Data loading and processing |
| `src.utils` | Metrics, featurization, and visualization |

## Quick Reference

### Main Classes

| Class | Module | Description |
|-------|--------|-------------|
| `VAEModule` | `src.vae_module` | Lightning module for VAE |
| `LDMModule` | `src.ldm_module` | Lightning module for LDM |
| `RLModule` | `src.rl_module` | Lightning module for RL |
| `DataModule` | `src.data` | Lightning DataModule |
| `CrystalBatch` | `src.data.schema` | Batch container for crystals |
| `Metrics` | `src.utils.metrics` | Evaluation metrics |

### Training Scripts

| Script | Description |
|--------|-------------|
| `src/train_vae.py` | Train VAE model |
| `src/train_ldm.py` | Train LDM model |
| `src/train_rl.py` | Train RL model |
| `src/train_predictor.py` | Train property predictor |
| `src/sample.py` | Generate structures |
| `src/evaluate.py` | Evaluate generated structures |

### Reward Components

| Component | Description |
|-----------|-------------|
| `CustomReward` | User-defined reward function |
| `PredictorReward` | Surrogate model predictions |
| `CreativityReward` | Uniqueness + Novelty |
| `EnergyReward` | Energy above hull minimization |
| `StructureDiversityReward` | MMD-based structure diversity |
| `CompositionDiversityReward` | Composition diversity |

## Usage Examples

### Load Pre-trained Model

```python
from src.vae_module import VAEModule
from src.ldm_module import LDMModule

# From HuggingFace Hub
vae = VAEModule.load_from_checkpoint("hspark1212/chemeleon2_mp_20_vae")
ldm = LDMModule.load_from_checkpoint("hspark1212/chemeleon2_mp_20_ldm")
```

### Generate Structures

```python
from src.sample import sample_structures

structures = sample_structures(
    vae_ckpt="hspark1212/chemeleon2_mp_20_vae",
    ldm_ckpt="hspark1212/chemeleon2_mp_20_ldm",
    num_samples=100,
)
```

### Evaluate Structures

```python
from src.utils.metrics import Metrics

metrics = Metrics(reference_structures=train_data)
results = metrics.compute(generated_structures)

print(f"Uniqueness: {results['unique']:.2%}")
print(f"Novelty: {results['novel']:.2%}")
```

### Custom Reward

```python
from src.rl_module.components import CustomReward

def my_reward_fn(batch_gen, **kwargs):
    # Custom reward logic
    return rewards

reward = CustomReward(reward_fn=my_reward_fn)
```

## Configuration

All modules use [Hydra](https://hydra.cc/) for configuration management.

```bash
# Override config via CLI
python src/train_vae.py experiment=mp_20/vae_dng trainer.max_epochs=100

# Show full config
python src/train_vae.py experiment=mp_20/vae_dng --cfg job
```

See `configs/` directory for all available configurations.
