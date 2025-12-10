# Tutorial: Predictor-Based Reward

Learn to train a property predictor and use it as a reward signal for RL fine-tuning.

## Overview

This tutorial covers the complete workflow:
1. Prepare a dataset with property labels
2. Train a property predictor
3. Configure PredictorReward for RL
4. Run RL training with property optimization

**Use case:** Optimize generated structures for band gap ~3.0 eV (wide band gap semiconductors).

## Prerequisites

- Pre-trained VAE checkpoint
- Pre-trained LDM checkpoint
- Dataset with property labels (e.g., band gap)

## Step 1: Prepare Your Dataset

### Option A: Use Existing Dataset

The Alex-MP-20-Bandgap dataset is ready to use:

```bash
ls data/alex_mp_20_bandgap/
# train.csv  val.csv  test.csv
```

### Option B: Create Custom Dataset

Query Materials Project for structures with band gap labels:

```python
import pandas as pd
from mp_api.client import MPRester

with MPRester(api_key="YOUR_API_KEY") as mpr:
    docs = mpr.materials.summary.search(
        num_sites=[1, 50],
        energy_above_hull=[0, 0.1],  # Stable structures
        band_gap=[0.1, None],         # Non-zero band gap
        fields=["material_id", "structure", "band_gap"],
    )

data = []
for doc in docs:
    data.append({
        "material_id": doc.material_id,
        "band_gap": doc.band_gap,
        "cif": doc.structure.to(fmt="cif"),
    })

df = pd.DataFrame(data)
print(f"Collected {len(df)} structures")

# Compute normalization statistics (needed for config)
print(f"Band gap mean: {df['band_gap'].mean():.3f}")
print(f"Band gap std: {df['band_gap'].std():.3f}")

# Split into train/val/test
train_df = df.sample(frac=0.8, random_state=42)
remaining = df.drop(train_df.index)
val_df = remaining.sample(frac=0.5, random_state=42)
test_df = remaining.drop(val_df.index)

# Save
train_df.to_csv("data/my_bandgap/train.csv", index=False)
val_df.to_csv("data/my_bandgap/val.csv", index=False)
test_df.to_csv("data/my_bandgap/test.csv", index=False)
```

## Step 2: Train the Predictor

### Create Predictor Configuration

Create `configs/experiment/my_bandgap/predictor.yaml`:

```yaml
# @package _global_
# Predictor training for band gap

data:
  _target_: src.data.datamodule.DataModule
  data_dir: ${paths.data_dir}/my_bandgap
  batch_size: 256
  dataset_type: "my_bandgap"
  target_condition: band_gap
  num_workers: 16

predictor_module:
  vae:
    checkpoint_path: ${hub:mp_20_vae}

  target_conditions:
    band_gap:
      mean: 1.5   # From Step 1 statistics
      std: 1.2    # From Step 1 statistics

logger:
  wandb:
    name: "predictor_bandgap"
```

### Run Predictor Training

```bash
python src/train_predictor.py experiment=my_bandgap/predictor
```

### Verify Predictor Quality

Check validation loss in WandB. The MAE should be reasonable for band gap prediction (typically < 0.5 eV).

## Step 3: Configure RL with Predictor Reward

### Understanding PredictorReward

The `PredictorReward` component (see [`src/rl_module/components.py:PredictorReward`](https://github.com/hspark1212/chemeleon2/blob/main/src/rl_module/components.py)) uses your trained predictor:

```python
class PredictorReward(RewardComponent):
    def __init__(
        self,
        predictor: PredictorModule,
        target_name: str,           # Which property to use
        target_value: float | None, # Optional: optimize toward this value
        clip_max: float | None,     # Optional: bound predictions
        clip_min: float | None,
        **kwargs,
    ):
        ...

    def compute(self, batch_gen: CrystalBatch, **kwargs) -> torch.Tensor:
        pred_val = self.predictor.predict(batch_gen)[self.target_name]

        if self.target_value is not None:
            # Negative squared error: closer to target = higher reward
            reward = -((pred_val - self.target_value) ** 2)
        else:
            # No target: maximize predicted value
            reward = pred_val

        return reward
```

### Create RL Configuration

Create `configs/experiment/my_bandgap/rl.yaml`:

```yaml
# @package _global_
# RL training for band gap optimization

data:
  _target_: src.data.datamodule.DataModule
  data_dir: ${paths.data_dir}/my_bandgap
  batch_size: 5
  dataset_type: "my_bandgap"
  target_condition: band_gap
  num_workers: 16

trainer:
  max_steps: 1000

rl_module:
  ldm_ckpt_path: ${hub:mp_20_ldm_base}
  vae_ckpt_path: ${hub:mp_20_vae}

  rl_configs:
    clip_ratio: 0.0001
    kl_weight: 1.0
    num_group_samples: 64
    group_reward_norm: true

  reward_fn:
    _target_: src.rl_module.reward.ReinforceReward
    normalize_fn: std
    eps: 1e-4
    reference_dataset: mp-20
    components:
      - _target_: src.rl_module.components.PredictorReward
        weight: 1.0
        normalize_fn: norm
        predictor:
          _target_: src.vae_module.predictor_module.PredictorModule.load_from_checkpoint
          checkpoint_path: "ckpts/my_bandgap/predictor.ckpt"
          map_location: "cpu"
        target_name: band_gap
        target_value: 3.0    # Target: wide band gap (3.0 eV)
        clip_min: 0.0        # Band gap cannot be negative

logger:
  wandb:
    name: rl_bandgap
```

## Step 4: Run RL Training

```bash
# Run RL with predictor reward (src/train_rl.py)
python src/train_rl.py experiment=my_bandgap/rl
```

Training script: [`src/train_rl.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/train_rl.py)

## Advanced Configuration

### Maximize vs. Target Value

**Maximize band gap** (no upper bound):
```yaml
- _target_: src.rl_module.components.PredictorReward
  target_name: band_gap
  target_value: null  # No target = maximize
  clip_min: 0.0
```

**Target specific value** (e.g., 2.5 eV):
```yaml
- _target_: src.rl_module.components.PredictorReward
  target_name: band_gap
  target_value: 2.5  # Penalize deviation from 2.5 eV
```

### Multi-Objective with Predictor

Combine band gap optimization with stability and diversity:

```yaml
components:
  - _target_: src.rl_module.components.PredictorReward
    weight: 1.0
    normalize_fn: norm
    predictor:
      _target_: src.vae_module.predictor_module.PredictorModule.load_from_checkpoint
      checkpoint_path: "ckpts/predictor.ckpt"
      map_location: "cpu"
    target_name: band_gap
    target_value: 3.0
    clip_min: 0.0

  - _target_: src.rl_module.components.EnergyReward
    weight: 0.5
    normalize_fn: norm

  - _target_: src.rl_module.components.CompositionDiversityReward
    weight: 0.5
    normalize_fn: norm
```

### Multiple Properties

Train predictor for multiple targets:

```yaml
# In predictor config
predictor_module:
  target_conditions:
    band_gap:
      mean: 1.5
      std: 1.2
    formation_energy:
      mean: -0.5
      std: 0.3
```

Use multiple PredictorReward components:

```yaml
components:
  - _target_: src.rl_module.components.PredictorReward
    weight: 1.0
    predictor:
      _target_: src.vae_module.predictor_module.PredictorModule.load_from_checkpoint
      checkpoint_path: "ckpts/predictor.ckpt"
      map_location: "cpu"
    target_name: band_gap
    target_value: 3.0

  - _target_: src.rl_module.components.PredictorReward
    weight: 0.5
    predictor:
      _target_: src.vae_module.predictor_module.PredictorModule.load_from_checkpoint
      checkpoint_path: "ckpts/predictor.ckpt"
      map_location: "cpu"
    target_name: formation_energy
    target_value: null  # Minimize (more negative = more stable)
```

## Troubleshooting

### Predictor Not Loading

Ensure checkpoint path and map_location are correct:

```yaml
predictor:
  _target_: src.vae_module.predictor_module.PredictorModule.load_from_checkpoint
  checkpoint_path: "ckpts/predictor.ckpt"  # Absolute or relative path
  map_location: "cpu"  # or "cuda:0" for GPU
```

### Reward Not Improving

1. **Check predictor quality**: Validation MAE should be reasonable
2. **Lower KL weight**: Allow more exploration
3. **Increase num_group_samples**: More stable gradient estimates
4. **Check target_value**: Ensure it's achievable

### Out of Memory

- Reduce `batch_size`
- Reduce `num_group_samples`
- Use `map_location: "cpu"` for predictor

## Reference Configuration

Working example: [`configs/experiment/alex_mp_20_bandgap/rl_bandgap.yaml`](https://github.com/hspark1212/chemeleon2/blob/main/configs/experiment/alex_mp_20_bandgap/rl_bandgap.yaml)

## Summary

1. **Prepared dataset** with property labels
2. **Trained predictor** in VAE latent space
3. **Configured PredictorReward** with target value
4. **Combined with other rewards** for robust optimization

The predictor-based approach enables efficient property optimization without expensive DFT calculations during training.

## Next Steps

- [DNG Reward](dng-reward.md) - Multi-objective optimization
- [Atomic Density](atomic-density.md) - Simple custom reward
- [Training Overview](../training/index.md) - General training guide
