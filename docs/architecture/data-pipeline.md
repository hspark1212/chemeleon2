# Data Pipeline

The data module (`src/data/`) handles loading, processing, and batching of crystal structure data.

## Data Flow

```{mermaid}
flowchart LR
    A[CIF/JSON Files] --> B[Dataset]
    B --> C[Featurizer]
    C --> D[CrystalBatch]
    D --> E[DataLoader]
    E --> F[Model]
```

## Key Classes

### CrystalBatch

The core data container for batched crystal structures:

```python
from src.data.schema import CrystalBatch

# CrystalBatch contains:
# - atom_types: Tensor of atomic numbers
# - frac_coords: Fractional coordinates
# - lengths: Lattice vector lengths
# - angles: Lattice angles
# - num_atoms: Number of atoms per structure
# - batch: Batch indices
```

### DataModule

PyTorch Lightning DataModule for training:

```python
from src.data import DataModule

datamodule = DataModule(
    data_path="data/mp-20",
    batch_size=32,
    num_workers=4,
)
```

### Featurizer

Converts pymatgen structures to model-ready features:

```python
from src.utils.featurizer import Featurizer

featurizer = Featurizer(
    max_atoms=20,
    atom_types=["H", "C", "N", "O", ...],
)

features = featurizer(structure)
```

## Supported Datasets

| Dataset | Description | Path |
|---------|-------------|------|
| MP-20 | Materials Project (≤20 atoms) | `data/mp-20/` |
| MP-120 | Materials Project (≤120 atoms) | `data/mp-120/` |
| Alex-MP-20 | Alexandria dataset (≤20 atoms) | `data/alex-mp-20/` |
| Amorphous | Amorphous materials | `data/mp_amorphous/` |

## Data Format

Crystal structures are stored as compressed JSON:

```python
from monty.serialization import loadfn

# Load dataset
structures = loadfn("data/mp-20/train.json.gz")

# Each entry contains:
# - structure: pymatgen Structure
# - material_id: MP identifier
# - formation_energy: Formation energy (optional)
```

## Configuration

See `configs/data/` for data configurations:

```yaml
# configs/data/mp_20.yaml
_target_: src.data.DataModule
data_path: data/mp-20
batch_size: 32
num_workers: 4
max_atoms: 20
train_val_test_split: [0.8, 0.1, 0.1]
```

## Utilities

### Metrics

Comprehensive evaluation metrics:

```python
from src.utils.metrics import Metrics

metrics = Metrics(reference_structures=train_data)
results = metrics.compute(generated_structures)
```

Available metrics:
- `unique` - Uniqueness rate
- `novel` - Novelty rate
- `e_above_hull` - Energy above hull
- `composition_validity` - Valid compositions
- `structure_diversity` - Structural diversity
- `composition_diversity` - Compositional diversity

### Visualization

3D structure visualization:

```python
from src.utils.visualize import visualize_structure

view = visualize_structure(structure)
view.show()
```

## Learn More

- [Evaluation Guide](../user-guide/evaluation.md) - Detailed metrics explanation
- [API Reference](../api/index.md) - Full API documentation
