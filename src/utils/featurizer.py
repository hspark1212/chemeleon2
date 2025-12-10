"""Structure featurization utilities for converting crystals to graphs."""

import torch
from pymatgen.core import Structure

from src.data.dataset_util import pmg_structure_to_pyg_data
from src.data.schema import CrystalBatch
from src.utils.checkpoint import get_checkpoint
from src.utils.scatter import scatter_mean, scatter_sum
from src.vae_module.vae_module import VAEModule

DEFAULT_MODEL_PATH = get_checkpoint("mp_20_vae")  # Auto-download from HF Hub


def featurize(
    structures: list[Structure],
    model_path: str | None = None,
    batch_size: int = 2000,
    use_encoder_features: bool = False,
    reduce: str = "mean",
    device: str | None = None,
):
    """Featurize a list of pymatgen Structures using a pre-trained VAE model.

    :param structures: List of pymatgen Structure objects to featurize.
    :param model_path: Path to the pre-trained VAE model checkpoint, defaults to DEFAULT_MODEL_PATH
    :param batch_size: Number of structures to process in each batch, defaults to 2000
    :param use_encoder_features: Whether to include encoder features along with latent vectors, defaults to False
    :param reduce: Reduction method for atom features to structure features ("mean" or "sum"), defaults to "mean"
    :param device: Device to use for featurization ("cuda" or "cpu").
        If None, automatically detects CUDA availability, defaults to None
    :return: Dictionary containing "structure_features" (torch.Tensor) and "atom_features" (list of torch.Tensor)
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pre-trained VAE
    if model_path is None:
        vae = VAEModule.load_from_checkpoint(
            DEFAULT_MODEL_PATH, map_location=device, weights_only=False
        )
    else:
        vae = VAEModule.load_from_checkpoint(
            model_path, map_location=device, weights_only=False
        )
    vae.eval()

    # Featurize each structure
    batch_size = min(batch_size, len(structures))
    structure_features = []
    composition_features = []
    atom_features = []

    reduce_fn = scatter_mean if reduce == "mean" else scatter_sum
    for i in range(0, len(structures), batch_size):
        batch_structures = structures[i : i + batch_size]
        batch = CrystalBatch.from_data_list(
            [pmg_structure_to_pyg_data(s) for s in batch_structures]
        )
        batch = batch.to(device)
        with torch.no_grad():
            encoded = vae.encode(batch)
            latent_vector = encoded["posterior"].mode()
            if use_encoder_features:
                latent_vector = torch.cat([latent_vector, encoded["x"]], dim=-1)
            composition_vector = vae.encoder.atom_type_embedder(batch.atom_types)  # type: ignore

        structure_feature = reduce_fn(latent_vector, batch.batch, dim=0)
        composition_feature = reduce_fn(composition_vector, batch.batch, dim=0)
        structure_features.append(structure_feature)
        composition_features.append(composition_feature)
        atom_features.extend(
            [
                latent_vector[batch.batch == i].detach().cpu()
                for i in range(batch.num_graphs)
            ]
        )
    structure_features = torch.cat(structure_features, dim=0).detach().cpu()
    composition_features = torch.cat(composition_features, dim=0).detach().cpu()
    out = {
        "structure_features": structure_features,
        "composition_features": composition_features,
        "atom_features": atom_features,
    }
    return out
