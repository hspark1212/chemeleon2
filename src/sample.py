"""Sampling script for generating crystal structures from trained models."""

import warnings
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from monty.serialization import dumpfn
from pymatgen.core import Composition, Structure

from src.data.num_atom_distributions import NUM_ATOM_DISTRIBUTIONS
from src.data.schema import create_empty_batch
from src.ldm_module.ldm_module import LDMModule
from src.paths import DEFAULT_LDM_CKPT_PATH, DEFAULT_VAE_CKPT_PATH


def sample(
    num_samples: int = 10000,
    batch_size: int = 2000,
    compositions: list | None = None,  # TODO: add feature in future
    text_prompts: list | None = None,  # TODO: add feature in future
    num_atom_distribution: str = "mp-20",
    ldm_ckpt_path: str | Path | None = None,
    vae_ckpt_path: str | Path | None = None,
    output_dir: str = "outputs",
    sampler: str = "ddim",
    sampling_steps: int = 50,
    cfg_scale: float = 2.0,  # Only used if use_cfg=True
    device: str | None = None,
    save_json: bool = True,
):
    r"""Sample crystal structures using a pre-trained LDM model.

    if compositions are provided, it performs the CSP (Crystal Structure Prediction) task.
    elif text_prompts are provided, it performs the TSP (Text-to-Structure Prediction) task.
    If neither compositions nor text_prompts are provided, it performs the DNG (De Novo Generation) task.

    :param num_samples: Total number of samples to generate, defaults to 10000
    :param batch_size: Number of samples to generate in each batch, defaults to 2000
    :param compositions: List of compositions to generate structures for.
        If provided, performs the CSP task, defaults to None
    :param text_prompts: List of text prompts to generate structures for.
        If provided, performs the TSP task, defaults to None
    :param num_atom_distribution: Name of the atom number distribution to use (e.g., "mp-20").
    :param ldm_ckpt_path: Path to the pre-trained LDM model checkpoint.
        If None, uses `DEFAULT_LDM_CKPT_PATH`, defaults to None
    :param vae_ckpt_path: Path to the pre-trained VAE model checkpoint.
        If None, uses `DEFAULT_VAE_CKPT_PATH`, defaults to None
    :param output_dir: Directory to save the generated CIF files and JSON summary,
        defaults to "outputs"
    :param sampler: Sampler to use for generating samples ("ddpm" or "ddim"), defaults to "ddim"
    :param sampling_steps: Number of sampling steps to use, defaults to 50
    :param cfg_scale: Classifier-free guidance scale, only used if use_cfg=True, defaults to 2.0
    :param device: Device to use for sampling ("cuda" or "cpu").
        If None, automatically detects CUDA availability, defaults to None
    :param save_json: Whether to save all generated structures in a single JSON.gz file,
        defaults to True

    :return: List of generated structures as ASE Atoms objects

    Examples\n
    --------
    CSP task
    # Generate 10 samples for each of the specified compositions (LiFePO4, Li2Co2O4, LiMn2O4, LiNiO2) (total 40 samples)
    >>> python src/sample.py --num_samples=10 --compositions="LiFePO4,Li2Co2O4,LiMn2O4,LiNiO2"

    DNG task
    # Generate 1000 samples using the 'mp-20' atom distribution
    >>> python src/sample.py --num_samples=1000 --batch_size=500 --num_atom_distributions="mp-20"

    # Using DDIM sampler with 50 sampling steps
    >>> python src/sample.py --num_samples=1000 --batch_size=1000 --sampler=ddim --sampling_steps=50
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Set default checkpoint path if not provided
    if ldm_ckpt_path is None and vae_ckpt_path is None:
        ldm_ckpt_path = DEFAULT_LDM_CKPT_PATH
        vae_ckpt_path = DEFAULT_VAE_CKPT_PATH
    assert ldm_ckpt_path is not None, "LDM checkpoint path must be provided."

    # Load the model
    if vae_ckpt_path is not None:
        ldm_module = LDMModule.load_from_checkpoint(
            ldm_ckpt_path, vae_ckpt_path=vae_ckpt_path, map_location=device
        )
    else:
        ldm_module = LDMModule.load_from_checkpoint(ldm_ckpt_path, map_location=device)
    ldm_module.eval()
    print(f"Loaded model from {ldm_ckpt_path}")

    # CSP task
    if compositions is not None:
        compositions = [Composition(c) for c in compositions]
        print(f"CSP task: Using {len(compositions)} compositions.")
        compositions = [c for c in compositions for _ in range(num_samples)]
        num_atoms = [int(c.num_atoms) for c in compositions]
    # DNG task
    else:
        num_atom_dist_dict = NUM_ATOM_DISTRIBUTIONS[num_atom_distribution]
        num_atoms = np.random.choice(
            list(num_atom_dist_dict.keys()),
            p=list(num_atom_dist_dict.values()),
            size=num_samples,
        ).tolist()
        print(f"DNG task: {num_samples} samples")
    # Set default batch size
    total_num_samples = len(num_atoms)
    batch_size = min(batch_size, total_num_samples)

    # Set output directory
    output_path = Path(output_dir)
    if output_path.exists():
        # Find any cif files in the output directory
        existing_cif_files = list(output_path.glob("sample_*.cif"))
        if existing_cif_files:
            warnings.warn(
                f"Output directory '{output_path}' already exists and contains {len(existing_cif_files)} CIF files. "
                "New samples will be generated alongside existing files.",
                UserWarning,
            )
    else:
        output_path.mkdir(parents=True, exist_ok=True)
    print(f"The sampled cif files will be saved in directory: '{output_path}'")

    # Sample
    sampled_structures = []
    for i in range(0, total_num_samples, batch_size):
        print(f"Generating batch #{i // batch_size + 1} with {batch_size} samples.")
        # Create an empty batch with the specified number of atoms
        batch = create_empty_batch(num_atoms[i : i + batch_size], device=device)
        # batch.y type annotation is dict[str, Tensor] but accepts list[Composition] at runtime
        batch.y = compositions[i : i + batch_size] if compositions else None  # type: ignore[assignment]
        # Generate samples
        with torch.no_grad():
            gen_st_list = ldm_module.sample(
                batch,
                sampler=sampler,
                sampling_steps=sampling_steps,
                cfg_scale=cfg_scale,
                return_structure=True,
            )
        # pymatgen Structure has to_ase_atoms() but not in type stubs
        sampled_structures.extend([st.to_ase_atoms() for st in gen_st_list])  # type: ignore[attr-defined]

        # Save generated structures
        assert isinstance(gen_st_list, list)
        for j, st in enumerate(gen_st_list):
            st.to(  # type: ignore[attr-defined]
                filename=str(
                    output_path / f"sample_{i + j}_{st.formula.replace(' ', '')}.cif"  # type: ignore[attr-defined]
                ),
                fmt="cif",
            )

    # Save generated structures in JSON format
    if save_json:
        gen_st_files = list(output_path.glob("sample_*.cif"))
        all_gen_st_list = [Structure.from_file(file) for file in gen_st_files]
        dumpfn(all_gen_st_list, output_path / "generated_structures.json.gz")
        print(
            f"The {len(all_gen_st_list)} generated structures saved in JSON format at: {output_path / 'generated_structures.json.gz'}"
        )
    return sampled_structures


def composition_to_atomic_numbers(composition):
    return [el.Z for el, amt in composition.items() for _ in range(int(amt))]


if __name__ == "__main__":
    Fire(sample)
