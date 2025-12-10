# Tutorial: Atomic Density Reward

Learn to create a custom reward that maximizes atomic density in generated crystals.

## Objective

Create a reward function that encourages denser crystal structures:

$$\text{density} = \frac{N_{\text{atoms}}}{V_{\text{cell}}}$$

Higher density = more atoms packed per unit volume.

## Prerequisites

- Pre-trained LDM checkpoint (or use `${hub:mp_20_ldm_base}`)
- Pre-trained VAE checkpoint (or use `${hub:mp_20_vae}`)

## Step 1: Understand the CustomReward Class

The `CustomReward` class in [`src/rl_module/components.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/rl_module/components.py) is a placeholder for user-defined logic:

```python
class CustomReward(RewardComponent):
    """Wrapper for user-defined custom reward functions."""

    def compute(self, gen_structures: list[Structure], **kwargs) -> torch.Tensor:
        """Placeholder for custom reward function."""
        return torch.zeros(len(gen_structures))
```

The `compute()` method receives:
- `gen_structures`: List of pymatgen `Structure` objects
- Additional kwargs like `batch_gen`, `device`, `metrics_obj`

## Step 2: Implement Atomic Density Reward

Edit [`src/rl_module/components.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/rl_module/components.py) and modify the `CustomReward` class:

```python
class CustomReward(RewardComponent):
    """Atomic density reward - maximize atoms per unit volume."""

    def compute(self, gen_structures: list[Structure], **kwargs) -> torch.Tensor:
        """
        Compute atomic density for each structure.

        Returns higher rewards for denser structures.
        """
        rewards = []
        for structure in gen_structures:
            num_atoms = len(structure)
            volume = structure.lattice.volume  # in Å³
            density = num_atoms / volume  # atoms per Å³
            rewards.append(density)
        return torch.tensor(rewards, dtype=torch.float32)
```

## Step 3: Create Configuration File

Create `configs/experiment/my_atomic_density.yaml`:

```yaml
# @package _global_
# Atomic Density RL Experiment

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
      - _target_: src.rl_module.components.CustomReward
        weight: 1.0
        normalize_fn: norm  # Scale to [0, 1] within batch

logger:
  wandb:
    name: rl_atomic_density
```

## Step 4: Run Training

```bash
# Run with custom config (src/train_rl.py)
python src/train_rl.py experiment=my_atomic_density
```

Training script: [`src/train_rl.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/train_rl.py)

## Step 5: Monitor Training

In WandB, watch these metrics:
- `train/reward`: Overall reward (should increase)
- `val/reward`: Validation reward

As training progresses, the model should generate increasingly dense structures.

## Extensions

### Target Density

Instead of maximizing density, optimize toward a specific target:

```python
class CustomReward(RewardComponent):
    """Target density reward - optimize toward specific density."""

    def __init__(self, target_density: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.target_density = target_density

    def compute(self, gen_structures: list[Structure], **kwargs) -> torch.Tensor:
        rewards = []
        for structure in gen_structures:
            density = len(structure) / structure.lattice.volume
            # Negative distance from target (higher = closer to target)
            reward = -abs(density - self.target_density)
            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32)
```

Config:
```yaml
components:
  - _target_: src.rl_module.components.CustomReward
    weight: 1.0
    target_density: 0.05  # atoms/Å³
    normalize_fn: std
```

### Combined with Stability

Ensure dense structures are also stable by adding `EnergyReward`:

```yaml
components:
  - _target_: src.rl_module.components.CustomReward
    weight: 1.0
    normalize_fn: norm
  - _target_: src.rl_module.components.EnergyReward
    weight: 0.5
    normalize_fn: norm
```

This balances density optimization with thermodynamic stability.

### Other Properties

The same pattern works for any structure property:

```python
# Example: Maximize number of unique elements
def compute(self, gen_structures: list[Structure], **kwargs) -> torch.Tensor:
    rewards = []
    for structure in gen_structures:
        num_elements = len(set(structure.species))
        rewards.append(float(num_elements))
    return torch.tensor(rewards, dtype=torch.float32)
```

```python
# Example: Target specific space group
def compute(self, gen_structures: list[Structure], **kwargs) -> torch.Tensor:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    target_spacegroup = 225  # Fm-3m (FCC)
    rewards = []
    for structure in gen_structures:
        try:
            sga = SpacegroupAnalyzer(structure)
            sg_number = sga.get_space_group_number()
            reward = 1.0 if sg_number == target_spacegroup else 0.0
        except Exception:
            reward = 0.0
        rewards.append(reward)
    return torch.tensor(rewards, dtype=torch.float32)
```

## Summary

1. Modify `CustomReward.compute()` in [`src/rl_module/components.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/rl_module/components.py)
2. Create experiment config with your reward component
3. Run training and monitor rewards
4. Combine with other components for multi-objective optimization

## Next Steps

- [DNG Reward](dng-reward.md) - Multi-objective optimization from the paper
- [Predictor Reward](predictor-reward.md) - Use ML models as reward
