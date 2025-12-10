# Predictor Training

The Predictor is an optional module that learns to predict material properties from VAE latent vectors. It enables property-based RL rewards without expensive property calculations during training.

## What Predictor Does

The Predictor operates in the VAE's latent space:

```
Crystal Structure → VAE Encoder → Latent z → Predictor → Property Value
```

Key benefits (see [`src/vae_module/predictor_module.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/vae_module/predictor_module.py)):
- **Fast inference**: Predicts from latent vectors, not raw structures
- **Differentiable**: Enables gradient-based optimization
- **Surrogate model**: Replaces expensive DFT/ML calculations

## Prerequisites

Predictor training requires:
1. **Trained VAE checkpoint**
2. **Dataset with property labels** (e.g., band gap, formation energy)

## Quick Start

```bash
# Train band gap predictor (src/train_predictor.py)
python src/train_predictor.py experiment=alex_mp_20_bandgap/predictor_dft_band_gap
```

Training script: [`src/train_predictor.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/train_predictor.py)
Example config: [`configs/experiment/alex_mp_20_bandgap/predictor_dft_band_gap.yaml`](https://github.com/hspark1212/chemeleon2/blob/main/configs/experiment/alex_mp_20_bandgap/predictor_dft_band_gap.yaml)

:::{tip}
**Example training run**: See [this W&B run](https://wandb.ai/hspark1212/chemeleon2/groups/train_predictor%2Falex_mp_20_bandgap/runs/e8mhs4yg) for a successful predictor training example for band gap on Alex MP-20 dataset.
:::

## Dataset Preparation

### Required Data Format

Your dataset CSV files need:
- `material_id`: Unique identifier
- `cif`: Crystal structure in CIF format
- Property column(s): e.g., `band_gap`, `formation_energy`

```
data/my_dataset/
├── train.csv
├── val.csv
└── test.csv
```

Example CSV:
```csv
material_id,cif,band_gap
mp-1234,"data_...",2.5
mp-5678,"data_...",0.0
```

### Compute Normalization Statistics

Calculate mean and std for your target property:

```python
import pandas as pd

df = pd.read_csv("data/my_dataset/train.csv")
print(f"band_gap mean: {df['band_gap'].mean():.3f}")
print(f"band_gap std: {df['band_gap'].std():.3f}")
```

These values are needed for the config.

## Training Commands

### Basic Training

```bash
# Use experiment config
python src/train_predictor.py experiment=alex_mp_20_bandgap/predictor_dft_band_gap

# Override parameters
python src/train_predictor.py experiment=alex_mp_20_bandgap/predictor_dft_band_gap \
    data.batch_size=512 \
    trainer.max_epochs=500
```

## Configuration

### Example Config

Create `configs/experiment/my_dataset/predictor_bandgap.yaml`:

```yaml
# @package _global_
# Predictor training for band gap

data:
  _target_: src.data.datamodule.DataModule
  data_dir: ${paths.data_dir}/my_dataset
  batch_size: 256
  dataset_type: "my_dataset"
  target_condition: band_gap
  num_workers: 16

predictor_module:
  vae:
    checkpoint_path: ${hub:mp_20_vae}

  target_conditions:
    band_gap:
      mean: 1.5   # From your dataset statistics
      std: 1.2    # From your dataset statistics

logger:
  wandb:
    name: "predictor_bandgap"
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 256 | Projection network hidden dimension |
| `num_layers` | 3 | Number of projection layers |
| `dropout` | 0.1 | Dropout rate |
| `use_encoder_features` | false | Concatenate encoder hidden features |

## Multiple Properties

Train a predictor for multiple properties simultaneously:

```yaml
predictor_module:
  target_conditions:
    band_gap:
      mean: 1.5
      std: 1.2
    formation_energy:
      mean: -0.5
      std: 0.3
```

## Training Tips

### Monitoring

Key metrics in WandB:
- `train/loss`: Overall MSE loss
- `train/band_gap_loss`: Per-property loss
- `val/loss`: Validation loss (check for overfitting)

### Typical Training

- **Duration**: ~200-500 epochs
- **Batch size**: 256-512
- **Early stopping**: Based on `val/loss`

### Verifying Quality

After training, check prediction quality:

```python
from src.vae_module.predictor_module import PredictorModule

predictor = PredictorModule.load_from_checkpoint(
    "ckpts/predictor.ckpt",
    map_location="cpu"
)
predictor.eval()

# Check validation MAE
# Should be reasonable for your property range
```

## Available Experiments

| Experiment | Dataset | Target | Description |
|------------|---------|--------|-------------|
| `alex_mp_20_bandgap/predictor_dft_band_gap` | Alex MP-20 | DFT band gap | Band gap prediction |

## Using Predictor for RL

After training, use the predictor as a reward signal:

```yaml
# In RL config
reward_fn:
  components:
    - _target_: src.rl_module.components.PredictorReward
      weight: 1.0
      predictor:
        _target_: src.vae_module.predictor_module.PredictorModule.load_from_checkpoint
        checkpoint_path: "ckpts/predictor.ckpt"
        map_location: "cpu"
      target_name: band_gap
      target_value: 3.0  # Optimize toward this value
```

See [Predictor Reward Tutorial](../rewards/predictor-reward.md) for the complete workflow.

## Next Steps

- [Predictor Reward Tutorial](../rewards/predictor-reward.md) - Complete RL workflow with predictor
- [RL Training](rl.md) - General RL training guide
