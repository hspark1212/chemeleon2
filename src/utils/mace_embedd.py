# ''' Calculatem MACE embeddings


# '''

# from mace.calculators import MACECalculator
# import h5py
# import pandas as pd
# from ase import Atoms
# from pymatgen.core import Structure
# from tqdm import tqdm



# def evaluate(
#     mace_model_path: str,
#     device: str = "cuda",
#     enable_cueq: bool = True,
# ):

#     calc = MACECalculator(model_path=mace_model_path, device=device, enable_cueq=enable_cueq)


#     with h5py.File("data/mp-20/mace_features.h5", "w") as f:
#         for split in ["train", "val", "test"]:
#             df = pd.read_csv(f"data/mp-20/{split}.csv")
#             for i, row in tqdm(df.iterrows(), total=len(df)):
#                 st = Structure.from_str(row["cif"], fmt="cif")
#                 atoms = st.to_ase_atoms()
#                 desc = calc.get_descriptors(atoms)
#                 material_id = row["material_id"]
#                 f.create_dataset(f"{material_id}", data=desc, compression="gzip")


# $ bash -lc cat <<'EOF' > src/utils/mace_embedd.py
# from __future__ import annotations

"""Compute and persist MACE descriptors for dataset splits."""

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from tqdm import tqdm

from mace.calculators import MACECalculator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_DATASET_DIR = DATA_ROOT / "mp-20"
DEFAULT_MODEL_PATH = Path(__file__).with_name("MACE-matpes-pbe-omat-ft.model")
DEFAULT_OUTPUT_NAME = "mace_features.h5"


def _resolve_device(device: str | None) -> str:
    """Return a valid device string, defaulting to CUDA when available."""
    if device:
        return device
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ModuleNotFoundError:
        return "cpu"


def _ensure_numpy(descriptor: Any) -> np.ndarray:
    """Convert descriptors returned by MACE to a NumPy array."""
    if isinstance(descriptor, dict):
        descriptor = next(iter(descriptor.values()))
    if isinstance(descriptor, np.ndarray):
        return descriptor
    if hasattr(descriptor, "detach"):
        return descriptor.detach().cpu().numpy()
    return np.asarray(descriptor)


def evaluate(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    mace_model_path: str | Path | None = None,
    output_path: str | Path | None = None,
    device: str | None = None,
    enable_cueq: bool = False,
    splits: tuple[str, ...] = ("train", "val", "test"),
    compression: str | None = "gzip",
    show_progress: bool = True,
) -> Path:
    """Compute MACE descriptors for a dataset and store them in an HDF5 file."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    if mace_model_path is None:
        mace_model_path = DEFAULT_MODEL_PATH
    mace_model_path = Path(mace_model_path)
    if not mace_model_path.exists():
        raise FileNotFoundError(f"MACE model checkpoint not found: {mace_model_path}")

    if output_path is None:
        output_path = dataset_dir / DEFAULT_OUTPUT_NAME
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_device = _resolve_device(device)
    print(f"Using device: {resolved_device}")
    print(f"Loading MACE model from: {mace_model_path}")
    calc = MACECalculator(
        model_path=str(mace_model_path),
        device=resolved_device,
        enable_cueq=enable_cueq,
    )

    splits = tuple(splits)
    with h5py.File(output_path, "w") as h5_file:
        for split in splits:
            csv_path = dataset_dir / f"{split}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing CSV for split '{split}': {csv_path}")
            df = pd.read_csv(csv_path)
            iterator = (
                tqdm(df.iterrows(), total=len(df), desc=f"{split} split")
                if show_progress
                else df.iterrows()
            )
            for _, row in iterator:
                structure = Structure.from_str(row["cif"], fmt="cif")
                atoms = structure.to_ase_atoms()
                descriptors = calc.get_descriptors(atoms)
                descriptor_array = _ensure_numpy(descriptors)
                material_id = str(row["material_id"])
                if material_id in h5_file:
                    # Skip duplicates while favouring the first occurrence.
                    continue
                h5_file.create_dataset(
                    material_id,
                    data=descriptor_array,
                    compression=compression,
                )

    print(f"Saved MACE descriptors to: {output_path}")
    return output_path


if __name__ == "__main__":
    import fire

    fire.Fire(evaluate)
