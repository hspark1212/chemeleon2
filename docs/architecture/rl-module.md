# RL Module

The Reinforcement Learning module (`src/rl_module/`) fine-tunes the LDM using Group Relative Policy Optimization (GRPO).

## Algorithm Overview

GRPO optimizes the LDM policy to maximize expected rewards:

$$\mathcal{L}_{GRPO} = -\mathbb{E}[\min(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)] + \beta D_{KL}$$

Where:
- $r_t(\theta)$: Probability ratio between current and old policy
- $A_t$: Advantage (normalized group rewards)
- $\epsilon$: Clipping parameter
- $\beta$: KL penalty weight

## Key Classes

### RLModule

PyTorch Lightning module for RL fine-tuning:

```python
from src.rl_module import RLModule

# Load RL module with pre-trained LDM
rl_module = RLModule(
    ldm_module=ldm,
    reward=reward_config,
)
```

**Key Methods:**
- `rollout(batch)` - Generates trajectories from LDM
- `compute_rewards(batch_gen)` - Evaluates generated structures
- `calculate_loss(...)` - Computes GRPO surrogate objective

### RewardComponent (Base Class)

Abstract base for all reward components:

```python
from src.rl_module.components import RewardComponent

class MyCustomReward(RewardComponent):
    def compute(self, batch_gen, **kwargs):
        # Return tensor of rewards for each structure
        return rewards
```

### Built-in Reward Components

| Component | Description | Key Parameter |
|-----------|-------------|---------------|
| `CustomReward` | User-defined rewards | `reward_fn` |
| `PredictorReward` | Surrogate model predictions | `predictor_ckpt` |
| `CreativityReward` | Unique + Novel structures | Uses metrics |
| `EnergyReward` | Low energy above hull | Uses MACE |
| `StructureDiversityReward` | MMD-based diversity | `sigma` |
| `CompositionDiversityReward` | Composition diversity | `sigma` |

### ReinforceReward

Aggregates multiple reward components:

```python
from src.rl_module.reward import ReinforceReward

reward = ReinforceReward(
    components=[
        {"name": "creativity", "weight": 1.0},
        {"name": "energy", "weight": 0.5},
    ],
    normalization="std",
)
```

## Normalization Strategies

| Strategy | Description |
|----------|-------------|
| `std` | Standardize to zero mean, unit variance |
| `norm` | Min-max normalization to [0, 1] |
| `subtract_mean` | Subtract mean only |
| `clip` | Clip to specified range |

## Configuration

See `configs/rl_module/` for RL configurations:

```yaml
# configs/rl_module/rl_dng.yaml
_target_: src.rl_module.RLModule
reward:
  components:
    - name: creativity
      weight: 1.0
    - name: energy
      weight: 0.5
  normalization: std
grpo:
  epsilon: 0.2
  kl_coef: 0.01
```

## Training

```bash
# De novo generation RL
python src/train_rl.py experiment=mp_20/rl_dng

# Band gap optimization RL
python src/train_rl.py experiment=alex_mp_20_bandgap/rl_bandgap
```

See [Training Guide](../user-guide/training.md) and [Custom Rewards](../user-guide/custom-rewards.md) for more details.
