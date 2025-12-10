# Tutorial: DNG (De Novo Generation) Reward

Learn to configure the multi-objective reward used in the Chemeleon2 paper for generating novel, stable, and diverse crystal structures.

## Overview

The DNG reward combines four complementary objectives:

| Component | Purpose | Weight |
|-----------|---------|--------|
| **CreativityReward** | Generate unique and novel structures | 1.0 |
| **EnergyReward** | Ensure thermodynamic stability | 1.0 |
| **StructureDiversityReward** | Explore varied crystal geometries | 0.1 |
| **CompositionDiversityReward** | Cover chemical composition space | 1.0 |

## Prerequisites

- Pre-trained LDM checkpoint
- Pre-trained VAE checkpoint
- MP-20 reference dataset (for novelty metrics, see [Evaluation Guide](../evaluation.md#prerequisites))

## The DNG Configuration

Reference file: [`configs/experiment/mp_20/rl_dng.yaml`](https://github.com/hspark1212/chemeleon2/blob/main/configs/experiment/mp_20/rl_dng.yaml)

```yaml
# @package _global_
# GRPO for DNG on MP-20

data:
  batch_size: 5

rl_module:
  ldm_ckpt_path: ${hub:mp_20_ldm_base}
  vae_ckpt_path: ${hub:mp_20_vae}

  rl_configs:
    num_group_samples: 64
    group_reward_norm: true

  reward_fn:
    _target_: src.rl_module.reward.ReinforceReward
    normalize_fn: std
    eps: 1e-4
    reference_dataset: mp-20
    components:
      - _target_: src.rl_module.components.CreativityReward
        weight: 1.0
        normalize_fn: null
      - _target_: src.rl_module.components.EnergyReward
        weight: 1.0
        normalize_fn: norm
      - _target_: src.rl_module.components.StructureDiversityReward
        weight: 0.1
        normalize_fn: norm
      - _target_: src.rl_module.components.CompositionDiversityReward
        weight: 1.0
        normalize_fn: norm

logger:
  wandb:
    name: rl_dng_grpo
```

## Component Deep Dive

All components are defined in [`src/rl_module/components.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/rl_module/components.py).

### CreativityReward

**Purpose:** Reward structures that are both unique (not duplicated in batch) and novel (not in training set) ([`src/rl_module/components.py:CreativityReward`](https://github.com/hspark1212/chemeleon2/blob/main/src/rl_module/components.py)).

**How it works:**

```python
for i, gen_structure in enumerate(gen_structures):
    u, v = metrics_results["unique"][i], metrics_results["novel"][i]
    if u and v:
        r = 1.0  # Fully creative: unique AND novel
    elif not u and not v:
        r = 0.0  # Not creative: duplicate of existing
    else:
        # Edge case: use AMD distance as continuous measure
        amds = structures_to_amd([gen_structure] + matching_refs, 100)
        dists = amd.AMD_cdist(amds, amds)[0]
        r = dists[dists > 0].min()
```

**Configuration:**
- `weight: 1.0` - Equal importance with other objectives
- `normalize_fn: null` - Already in [0, 1] range

### EnergyReward

**Purpose:** Penalize structures with high energy above the convex hull.

**How it works:**
- Computes formation energy using MACE-torch
- Compares to Materials Project convex hull
- Returns negative energy (minimization → maximization)

```python
r_energy = torch.as_tensor(metrics_results["e_above_hull"]).float()
r_energy = r_energy.nan_to_num(nan=1.0)  # Handle failed calculations
r_energy = r_energy.clamp(min=0.0, max=1.0)  # Clip to reasonable range
r_energy = r_energy * -1.0  # Negative for minimization
```

**Configuration:**
- `weight: 1.0` - Strong emphasis on stability
- `normalize_fn: norm` - Scale within batch

### StructureDiversityReward

**Purpose:** Encourage diverse crystal geometries using Maximum Mean Discrepancy (MMD).

**How it works:**
- Featurizes structures (lattice parameters, atomic positions)
- Computes MMD between generated batch and reference distribution
- Rewards structures that differ from existing patterns

**Configuration:**
- `weight: 0.1` - Lower weight prevents over-diversification
- `normalize_fn: norm` - Scale within batch

**Why lower weight?**
Too much structure diversity can lead to:
- Physically unrealistic geometries
- Sacrificing stability for novelty

### CompositionDiversityReward

**Purpose:** Encourage exploration of chemical composition space.

**How it works:**
- Extracts element-wise composition features
- Computes MMD between generated and reference compositions
- Rewards deviating from common compositions

**Configuration:**
- `weight: 1.0` - Strong emphasis on chemical diversity
- `normalize_fn: norm` - Scale within batch

## Running DNG Training

```bash
# Standard DNG training (src/train_rl.py)
python src/train_rl.py experiment=mp_20/rl_dng

# With custom hyperparameters
python src/train_rl.py experiment=mp_20/rl_dng \
    rl_module.rl_configs.num_group_samples=128 \
    trainer.max_steps=2000

# Override checkpoint paths
python src/train_rl.py experiment=mp_20/rl_dng \
    rl_module.ldm_ckpt_path=ckpts/my_ldm.ckpt
```

## Monitoring Training

Key WandB metrics:

| Metric | Expected Behavior |
|--------|-------------------|
| `train/reward` | Increase over time |
| `train/creativity` | Increase (more novel structures) |
| `train/energy` | Increase (lower energy = higher reward) |
| `train/structure_diversity` | Stable or slight increase |
| `train/composition_diversity` | Increase (broader chemical space) |

## Weight Tuning Guide

Adjust weights based on your priorities:

| Priority | CreativityReward | EnergyReward | StructureDiversity | CompositionDiversity |
|----------|------------------|--------------|--------------------|-----------------------|
| More novelty | ↑ 1.5 | ↓ 0.5 | 0.1 | 1.0 |
| More stability | 0.5 | ↑ 2.0 | 0.1 | 0.5 |
| More diversity | 1.0 | 0.5 | ↑ 0.5 | ↑ 1.5 |
| Balanced (default) | 1.0 | 1.0 | 0.1 | 1.0 |

```yaml
# Example: Prioritize stability
components:
  - _target_: src.rl_module.components.CreativityReward
    weight: 0.5
  - _target_: src.rl_module.components.EnergyReward
    weight: 2.0
    normalize_fn: norm
  - _target_: src.rl_module.components.StructureDiversityReward
    weight: 0.1
    normalize_fn: norm
  - _target_: src.rl_module.components.CompositionDiversityReward
    weight: 0.5
    normalize_fn: norm
```

## Customization Examples

### Add Property Constraint

Combine DNG with band gap optimization:

```yaml
components:
  - _target_: src.rl_module.components.CreativityReward
    weight: 0.5
  - _target_: src.rl_module.components.EnergyReward
    weight: 0.5
    normalize_fn: norm
  - _target_: src.rl_module.components.PredictorReward
    weight: 1.0
    normalize_fn: norm
    predictor:
      _target_: src.vae_module.predictor_module.PredictorModule.load_from_checkpoint
      checkpoint_path: "ckpts/predictor.ckpt"
      map_location: "cpu"
    target_name: band_gap
    target_value: 2.0
  - _target_: src.rl_module.components.CompositionDiversityReward
    weight: 0.5
    normalize_fn: norm
```

### Focus on Specific Elements

Use a custom reference dataset for specific chemistry:

```yaml
reward_fn:
  _target_: src.rl_module.reward.ReinforceReward
  reference_dataset: my_custom_dataset  # Your filtered dataset
  components:
    # ... components
```

### Remove Diversity Constraints

For focused optimization without diversity:

```yaml
components:
  - _target_: src.rl_module.components.CreativityReward
    weight: 1.0
  - _target_: src.rl_module.components.EnergyReward
    weight: 1.0
    normalize_fn: norm
  # No diversity rewards
```

## Summary

The DNG reward configuration:
1. **Balances multiple objectives** for well-rounded generation
2. **Prevents mode collapse** with diversity rewards
3. **Ensures physical validity** with energy penalty
4. **Encourages exploration** with creativity bonus

## Next Steps

- [Predictor Reward](predictor-reward.md) - Property-targeted optimization
- [Atomic Density](atomic-density.md) - Simple custom reward example
