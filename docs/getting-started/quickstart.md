# Quick Start

This guide walks you through generating crystal structures using pre-trained models.

## Download Pre-trained Models

Chemeleon2 provides pre-trained checkpoints via [HuggingFace Hub](https://huggingface.co/hspark1212/chemeleon2-checkpoints). Below is a list of available models:

| Model | Dataset | Description |
|-------|---------|-------------|
| `mp_20_vae` | MP-20 | pre-trained VAE trained on MP-20 dataset |
| `alex_mp_20_vae` | Alex-MP-20 | pre-trained VAE trained on Alexandria MP-20 dataset |
| `mp_20_ldm_base` | MP-20 | pre-trained LDM on MP-20 dataset |
| `alex_mp_20_ldm_base` | Alex-MP-20 | pre-trained LDM on Alexandria MP-20 dataset |
| `mp_20_ldm_rl`| MP-20 | Fine-tuned LDM with RL for DNG rewards on MP-20 dataset |
| `alex_mp_20_ldm_rl` | Alex-MP-20 | Fine-tuned LDM with RL for DNG rewards on Alexandria MP-20 dataset |

```{tip}
**Automatic Checkpoint Loading**

You don't need to manually download checkpoints! Use the `${hub:...}` resolver in config files or command-line arguments to automatically download from HuggingFace Hub. Available hub identifiers match the model names in the table above. See [Training Guide](../user-guide/training/index.md#checkpoint-management) for more details.
```

## Sample Crystal Structures

### Using Python API

- Generate 1000 structures using the Default model (`alex_mp_20_ldm_rl`)
```python
from src.sample import sample

# Sample 1000 structures with Default model trained with alex-mp-20
gen_atoms_list = sample(
    num_samples=1000, 
    batch_size=500, 
    output_dir="outputs/alex-mp-20",
)

# Each structure is an ASE Atoms object
from ase.visualize import view

view(gen_atoms_list[0], viewer="ngl")
```

### Using Command Line

- You can also run sampling with specific paths of VAE and LDM checkpoints via CLI:
```bash
python src/sample.py \
    --num_samples=100 \
    --vae_ckpt_path=checkpoints/v0.0.1/alex_mp_20/vae/dng_j1jgz9t0_v1.ckpt \
    --ldm_ckpt_path=checkpoints/v0.0.1/alex_mp_20/ldm/ldm_rl_dng_tuor5vgd.ckpt \
    --output_dir="outputs/alex-mp-20-cli"
```

## Evaluate Generated Structures

Calculate metastable Structure Uniqueness and Novelty (mSUN) to assess the quality of generated structures:

```python
from monty.serialization import loadfn
from src.utils.metrics import Metrics

# Load generated structures from previous step
gen_structures = loadfn("outputs/alex-mp-20/generated_structures.json.gz")

# Initialize metrics calculator
metrics = Metrics(
    metrics=["unique", "novel", "e_above_hull"],
    reference_dataset="mp-20",
    phase_diagram="mp-all",
    metastable_threshold=0.1,
)

# Compute metrics
results = metrics.compute(gen_structures=gen_structures)

# Calculate mSUN score (percentage that are unique, novel, AND metastable)
msun_score = (
    results["unique"] & results["novel"] & results["is_metastable"]
).mean() * 100
print(f"mSUN Score: {msun_score:.2f}%")
print(f"Uniqueness: {results['unique'].mean():.2%}")
print(f"Novelty: {results['novel'].mean():.2%}")
print(f"Metastable: {results['is_metastable'].mean():.2%}")

# Convert to DataFrame for analysis
df = metrics.to_dataframe()
df.head()

# Save results to CSV
df.to_csv("outputs/alex-mp-20/metrics_results.csv")
```

For detailed metrics documentation, see the [Evaluation Guide](../user-guide/evaluation.md).

## Interactive Tutorial

For a more detailed walkthrough, see the [tutorial.ipynb](https://github.com/hspark1212/chemeleon2/blob/main/tutorial.ipynb) notebook.

## Next Steps

- [Training Guide](../user-guide/training/index.md) - Train VAE, LDM, and RL models
- [Evaluation Guide](../user-guide/evaluation.md) - Detailed metrics explanation
- [Custom Rewards](../user-guide/rewards/index.md) - Define custom RL reward functions
