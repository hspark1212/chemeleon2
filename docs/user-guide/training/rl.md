# RL Training

Reinforcement Learning (RL) is the third stage of the Chemeleon2 pipeline. It fine-tunes the LDM to generate crystal structures that maximize user-defined reward functions.

## What RL Does

RL optimizes the generation process using reward signals:

```
LDM generates structures → Compute rewards → Update LDM to increase rewards
```

Key concepts (see [`src/rl_module/rl_module.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/rl_module/rl_module.py)):
- **GRPO Algorithm**: Group Relative Policy Optimization for efficient training
- **Reward Functions**: Define what properties to optimize (see [`src/rl_module/reward.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/rl_module/reward.py))
- **Policy Update**: Adjust LDM weights to favor high-reward structures

## Prerequisites

RL training requires:
1. **Trained LDM checkpoint**
2. **Trained VAE checkpoint** (usually loaded via LDM)

```yaml
rl_module:
  ldm_ckpt_path: ${hub:mp_20_ldm_base}
  vae_ckpt_path: ${hub:mp_20_vae}
```

## Quick Start

```bash
# Fine-tune with de novo generation reward (src/train_rl.py)
python src/train_rl.py experiment=mp_20/rl_dng
```

Training script: [`src/train_rl.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/train_rl.py)
Example config: [`configs/experiment/mp_20/rl_dng.yaml`](https://github.com/hspark1212/chemeleon2/blob/main/configs/experiment/mp_20/rl_dng.yaml)

:::{tip}
**Example training run**: See [this W&B run](https://wandb.ai/hspark1212/chemeleon2/groups/train_rl%2Fmp_20/runs/6xz1npuj) for a successful RL training example with DNG reward on MP-20 dataset.
:::

## Training Commands

### Basic Training

```bash
# Use experiment config
python src/train_rl.py experiment=mp_20/rl_dng

# Override checkpoint paths
python src/train_rl.py experiment=mp_20/rl_dng \
    rl_module.ldm_ckpt_path=ckpts/my_ldm.ckpt

# Override RL hyperparameters
python src/train_rl.py experiment=mp_20/rl_dng \
    rl_module.rl_configs.num_group_samples=128 \
    data.batch_size=8
```

## GRPO Algorithm

Chemeleon2 uses **Group Relative Policy Optimization (GRPO)** for efficient RL training:

1. **Sample Groups**: Generate multiple structures per batch
2. **Compute Rewards**: Evaluate all structures in the group
3. **Relative Ranking**: Compare rewards within each group
4. **Policy Update**: Reinforce high-reward structures relative to group

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_group_samples` | 64 | Structures per group (higher = more stable) |
| `group_reward_norm` | true | Normalize rewards within group |
| `clip_ratio` | 0.0001 | PPO-style clipping ratio |
| `kl_weight` | 1.0 | KL divergence penalty weight |

```bash
# Example: larger groups for more stable training
python src/train_rl.py experiment=mp_20/rl_dng \
    rl_module.rl_configs.num_group_samples=128
```

## Reward Configuration

Rewards are defined in the `reward_fn` section of the config (see [`configs/experiment/mp_20/rl_dng.yaml`](https://github.com/hspark1212/chemeleon2/blob/main/configs/experiment/mp_20/rl_dng.yaml)):

```yaml
rl_module:
  reward_fn:
    _target_: src.rl_module.reward.ReinforceReward
    normalize_fn: std           # Global normalization
    eps: 1e-4
    reference_dataset: mp-20    # For novelty/uniqueness metrics
    components:
      - _target_: src.rl_module.components.CreativityReward
        weight: 1.0
        normalize_fn: null
      - _target_: src.rl_module.components.EnergyReward
        weight: 1.0
        normalize_fn: norm
```

See [Custom Rewards Guide](../rewards/index.md) for detailed component documentation ([`src/rl_module/components.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/rl_module/components.py)).

## Available Experiments

| Experiment | Dataset | Reward | Description |
|------------|---------|--------|-------------|
| `mp_20/rl_dng` | MP-20 | DNG (multi-objective) | Paper's de novo generation |
| `alex_mp_20_bandgap/rl_bandgap` | Alex MP-20 | Predictor-based | Band gap optimization |

## Training Tips

### Monitoring

Key metrics to watch in WandB:
- `train/reward`: Average reward (should increase)
- `train/kl_div`: KL divergence from original LDM
- `val/reward`: Validation reward for generalization
- Component-specific metrics (e.g., `train/creativity`, `train/energy`)

### Hyperparameter Tuning

| Issue | Solution |
|-------|----------|
| Unstable training | Increase `num_group_samples`, enable `group_reward_norm` |
| Mode collapse | Increase `kl_weight`, add diversity rewards |
| Slow convergence | Decrease `kl_weight`, increase reward weights |
| Poor structure quality | Add `EnergyReward` component |

### Typical Training

- **Duration**: ~500-2000 steps
- **Batch size**: 4-8 (limited by `num_group_samples`)
- **GPU memory**: Scales with `num_group_samples`

## Next Steps

- [Custom Rewards Overview](../rewards/index.md) - Learn about reward components
- [DNG Reward Tutorial](../rewards/dng-reward.md) - Paper's multi-objective reward
- [Predictor Reward Tutorial](../rewards/predictor-reward.md) - Property optimization
