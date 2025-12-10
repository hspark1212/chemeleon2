# Custom Rewards Overview

This guide explains how to configure and customize reward functions for RL training in Chemeleon2.

## What Are Rewards?

Reward functions define what properties to optimize during RL fine-tuning. The LDM learns to generate structures that maximize the total reward.

```
Generated Structure → Reward Components → Total Reward → Policy Update
```

## Quick Decision Guide

Choose the tutorial based on your use case:

| Use Case | Tutorial | Description |
|----------|----------|-------------|
| Simple custom logic | [Atomic Density](atomic-density.md) | Modify `CustomReward` class |
| Multi-objective (paper) | [DNG Reward](dng-reward.md) | Creativity + stability + diversity |
| Property optimization | [Predictor Reward](predictor-reward.md) | Train predictor, use as reward |

## Built-in Reward Components

All components are in [`src/rl_module/components.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/rl_module/components.py):

| Component | Description | Required Metrics |
|-----------|-------------|------------------|
| `CustomReward` | User-defined reward function | None |
| `PredictorReward` | Property prediction from trained predictor | None |
| `CreativityReward` | Rewards unique and novel structures | `unique`, `novel` |
| `EnergyReward` | Penalizes high energy above convex hull | `e_above_hull` |
| `StructureDiversityReward` | Rewards diverse crystal geometries (MMD) | `structure_diversity` |
| `CompositionDiversityReward` | Rewards diverse chemical compositions (MMD) | `composition_diversity` |

## RewardComponent Base Class

All reward components inherit from `RewardComponent`:

```python
class RewardComponent(ABC, torch.nn.Module):
    def __init__(
        self,
        weight: float = 1.0,          # Relative importance
        normalize_fn: str | None = None,  # Normalization strategy
        eps: float = 1e-4,            # Numerical stability
    ):
        ...

    @abstractmethod
    def compute(self, **kwargs) -> torch.Tensor:
        """Compute raw reward values."""
        pass

    def forward(self, **kwargs) -> torch.Tensor:
        """Compute, normalize, and weight the reward."""
        rewards = self.compute(**kwargs)
        if self.normalize_fn:
            rewards = self._normalize(rewards)
        return rewards * self.weight
```

### Available kwargs in `compute()`

| Argument | Type | Description |
|----------|------|-------------|
| `gen_structures` | `list[Structure]` | Generated pymatgen Structure objects |
| `batch_gen` | `CrystalBatch` | Batched tensor representation |
| `metrics_obj` | `Metrics` | Pre-computed metrics (if `required_metrics` is set) |
| `device` | `torch.device` | Current device |

## Normalization Options

Each component can apply normalization via `normalize_fn`:

| Option | Formula | Use Case |
|--------|---------|----------|
| `norm` | `(x - min) / (max - min)` | Scale to [0, 1] |
| `std` | `(x - mean) / std` | Zero mean, unit variance |
| `subtract_mean` | `x - mean` | Center around zero |
| `clip` | `clamp(x, -1, 1)` | Bound extreme values |
| `null` | No change | Already normalized (e.g., CreativityReward) |

## ReinforceReward Aggregation

The `ReinforceReward` class (see [`src/rl_module/reward.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/rl_module/reward.py)) combines multiple components:

```yaml
reward_fn:
  _target_: src.rl_module.reward.ReinforceReward
  normalize_fn: std           # Global normalization after combining
  eps: 1e-4
  reference_dataset: mp-20    # For novelty/uniqueness metrics
  components:
    - _target_: src.rl_module.components.CreativityReward
      weight: 1.0
      normalize_fn: null
    - _target_: src.rl_module.components.EnergyReward
      weight: 0.5
      normalize_fn: norm
```

### How Rewards Are Combined

1. Each component computes its reward
2. Component-level normalization is applied (if specified)
3. Rewards are multiplied by weights
4. All weighted rewards are summed
5. Global normalization is applied (if specified)

```
Total = normalize_global( sum( weight_i * normalize_i(reward_i) ) )
```

## Component Details

### CustomReward

Placeholder for user-defined logic. Modify directly in [`src/rl_module/components.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/rl_module/components.py):

```python
class CustomReward(RewardComponent):
    def compute(self, gen_structures: list[Structure], **kwargs) -> torch.Tensor:
        # Your custom logic here
        rewards = [your_function(s) for s in gen_structures]
        return torch.tensor(rewards, dtype=torch.float32)
```

### CreativityReward

Rewards structures that are both unique (not duplicated in batch) and novel (not in training set):

- Returns 1.0 if unique AND novel
- Returns 0.0 if not unique AND not novel
- Uses AMD distance for edge cases

### EnergyReward

Penalizes high energy above the convex hull:

- Uses MACE-torch for energy calculations
- Clips energy to [0, 1] eV/atom
- Returns negative energy (lower is better)

### StructureDiversityReward

Encourages diverse crystal geometries using Maximum Mean Discrepancy (MMD):

- Compares generated structures to reference distribution
- Rewards structures that differ from existing ones
- Uses polynomial kernel for smooth gradients

### CompositionDiversityReward

Encourages diverse chemical compositions using MMD:

- Compares element distributions to reference
- Rewards exploration of chemical space

### PredictorReward

Uses a trained predictor as surrogate model:

```yaml
- _target_: src.rl_module.components.PredictorReward
  weight: 1.0
  predictor:
    _target_: src.vae_module.predictor_module.PredictorModule.load_from_checkpoint
    checkpoint_path: "ckpts/predictor.ckpt"
    map_location: "cpu"
  target_name: band_gap
  target_value: 3.0    # Optional: optimize toward this value
  clip_min: 0.0        # Optional: bound predictions
```

- If `target_value` is set: Returns `-(pred - target)²`
- If `target_value` is None: Returns `pred` (maximize)

## Tutorials

- [Atomic Density](atomic-density.md) - Simple custom reward example
- [DNG Reward](dng-reward.md) - Paper's multi-objective configuration
- [Predictor Reward](predictor-reward.md) - Property optimization workflow
