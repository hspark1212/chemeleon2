# Evaluation Guide

This guide covers evaluating generated crystal structures against reference datasets to assess quality and diversity.

(prerequisites)=
## Prerequisites

Before running evaluation metrics, you need to download and extract the reference dataset.

### Download from Figshare

You can download directly from the web:

[Download benchmarks_mp_20.tar.gz from Figshare](https://figshare.com/articles/dataset/Assets_for_implementation_benchmarking_in_Chemeleon2_repo_https_github_com_hspark1212_chemeleon2_/30589436?file=59462369)

Or use the command line (from project root):

```bash
# Download the reference dataset
curl -L -A "Mozilla/5.0" -o benchmarks_mp_20.tar.gz https://figshare.com/ndownloader/files/59462369

# Extract the dataset
tar -zxvf benchmarks_mp_20.tar.gz
```

This will create the following directory structure:

```plaintext
benchmarks/
└── assets/
    ├── mp_20_all_composition_features.pt
    ├── mp_20_all_structure_features.pt
    ├── mp_20_all_structure.json.gz
    ├── mp_all_unique_structure_250416.json.gz
    └── ppd-mp_all_entries_uncorrected_250409.pkl.gz
```

These files contain the reference data required for computing evaluation metrics against the MP-20 dataset.

(generate-samples)=
## Generate Samples

Generate crystal structures using a pre-trained LDM model. (Default model is trained on alex-mp-20 dataset.)

```bash
# Generate 10000 samples with 2000 batch size using DDIM sampler
python src/sample.py --num_samples=10000 --batch_size=2000 --output_dir=outputs/samples
```

(evaluate-models)=
## Evaluate Models

Evaluate generated structures against reference datasets (i.e., MP-20, Alex-MP-20) to assess quality and diversity.

### Generate and Evaluate Together

Generate new structures and evaluate them in one command:

```bash
python src/evaluate.py \
    --model_path=ckpts/mp_20/ldm/ldm_null.ckpt \
    --structure_path=outputs/eval_samples \
    --reference_dataset=mp-20 \
    --num_samples=10000 \
    --batch_size=2000
```

### Evaluate Pre-generated Structures

If you already have generated structures:

```bash
python src/evaluate.py \
    --structure_path=outputs/dng_samples \
    --reference_dataset=mp-20 \
    --output_file=benchmark/results/my_results.csv
```

(evaluation-metrics)=
## Evaluation Metrics

The evaluation script computes several metrics to assess generation quality:

- **Unique**: Identifies structures that are not duplicates within the generated set
- **Novel**: Identifies structures not found in the reference dataset
- **E Above Hull**: Calculates the energy above hull for each structure to assess thermodynamic stability (also computes Metastable/Stable)
- **Composition Validity**: Checks if the composition is chemically valid using SMACT
- **Structure Diversity**: Computes inverse Fréchet distance (1/(1+FMD)) between generated and reference structure embeddings from VAE (higher is better)
- **Composition Diversity**: Computes inverse Fréchet distance (1/(1+FMD)) between generated and reference composition embeddings from VAE (higher is better)
- **Synthesizability**: Predicts synthesizability using CL-score (optional)

For detailed implementation, see `src/utils/metrics.py`.

### Python API Usage

You can also compute metrics using the Python API directly:

```python
from monty.serialization import loadfn
from src.utils.metrics import Metrics

# Load generated structures
gen_structures = loadfn("outputs/eval_samples/structures.json.gz")

# Create metrics object
metrics = Metrics(
    metrics=["unique", "novel", "e_above_hull", "composition_validity"],
    reference_dataset="mp-20",
    phase_diagram="mp-all",
    metastable_threshold=0.1,
    progress_bar=True,
)

# Compute metrics
results = metrics.compute(gen_structures=gen_structures)

# Save results
metrics.to_csv("outputs/results.csv")

# Or get as DataFrame
df = metrics.to_dataframe()
print(df.head())
```

### Reference Datasets

Available reference datasets:

- `mp-20`: Materials Project structures with ≤20 atoms
- `alex-mp-20`: Alexandria MP structures with ≤20 atoms

Results are saved to the specified output file in CSV format for further analysis.

(benchmarks)=
## Benchmarks for Chemeleon2 DNG

Pre-computed benchmark results for de novo generation (DNG) are available in the `benchmarks/dng/` directory:

- **MP-20**: `benchmarks/dng/chemeleon2_rl_dng_mp_20.json.gz` - 10,000 generated structures using RL-trained model on MP-20
- **Alex-MP-20**: `benchmarks/dng/chemeleon2_rl_dng_alex_mp_20.json.gz` - 10,000 generated structures using RL-trained model on Alex-MP-20

### Loading Benchmark Data

These files contain generated crystal structures in compressed JSON format:

```python
from monty.serialization import loadfn

# Load benchmark structures
structures = loadfn("benchmarks/dng/chemeleon2_rl_dng_mp_20.json.gz")
print(f"Loaded {len(structures)} structures")
```
