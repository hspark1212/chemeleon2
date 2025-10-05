"""Data augmentation utilities.

Implementation inspired by the work of All-Atom Diffusion Transformer
(https://github.com/facebookresearch/all-atom-diffusion-transformer)
"""

import torch

from src.data.schema import CrystalBatch


def apply_augmentation(
    batch: CrystalBatch, translate: bool, rotate: bool
) -> CrystalBatch:
    # Create a copy of the batch for augmentation
    batch_aug = batch.clone()  # type: ignore

    # Apply translate
    if translate:
        batch_aug = _augmentation_translate(batch_aug)

    # Apply rotate
    if rotate:
        batch_aug = _augmentation_rotate(batch_aug)

    return batch_aug


def _augmentation_translate(batch: CrystalBatch) -> CrystalBatch:
    # Translate cartesian coordinates
    random_translate = (
        torch.normal(
            torch.abs(batch.lengths.mean(dim=0)),
            torch.abs(batch.lengths.std(dim=0)) + 1e-8,
        )
        / 2
    )
    cart_coords_aug = batch.cart_coords + random_translate

    # Convert to fractional coordinates
    cell_per_node_inv = torch.linalg.inv(batch.lattices[batch.batch])
    frac_coords_aug = torch.einsum("bi,bij->bj", cart_coords_aug, cell_per_node_inv)
    frac_coords_aug = frac_coords_aug % 1.0

    # Update the batch with augmented coordinates
    batch.update(
        cart_coords=cart_coords_aug,
        frac_coords=frac_coords_aug,
    )
    return batch


def _augmentation_rotate(batch: CrystalBatch) -> CrystalBatch:
    # Rotate cartesian coordinates and lattices (frac coords should be same)
    device = batch.cart_coords.device
    rot_mat = _random_rotation_matrix(validate=True, device=device)
    cart_coords_aug = batch.cart_coords @ rot_mat.T
    lattices_aug = batch.lattices @ rot_mat.T

    # Update the batch with augmented coordinates and lattices
    batch.update(
        cart_coords=cart_coords_aug,
        lattices=lattices_aug,
    )
    return batch


def apply_noise(
    batch: CrystalBatch,
    ratio: float,
    corruption_scale: float = 0.1,  # Å
) -> CrystalBatch:
    batch_noise = batch.clone()  # type: ignore
    device = batch_noise.cart_coords.device

    # Select indices to apply noise
    total_num_atoms = batch_noise.num_nodes
    noise_num_atoms = int(total_num_atoms * ratio)

    # Apply masking noise to atom types
    noise_atom_types = batch_noise.atom_types.clone()
    noise_indices = torch.randperm(total_num_atoms, device=device)[:noise_num_atoms]
    noise_atom_types[noise_indices] = 0

    # Apply random movement to cartesian coordinates
    noise_cart_coords = batch_noise.cart_coords.clone()
    noise_indices = torch.randperm(total_num_atoms, device=device)[:noise_num_atoms]
    noise_cart_coords[noise_indices] += (
        torch.randn((noise_num_atoms, 3), device=device) * corruption_scale
    )

    # Convert to fractional coordinates
    cell_per_node_inv = torch.linalg.inv(batch.lattices[batch.batch])
    noise_frac_coords = torch.einsum("bi,bij->bj", noise_cart_coords, cell_per_node_inv)
    noise_frac_coords = noise_frac_coords % 1.0

    # Update the batch with noise
    batch_noise.update(
        atom_types=noise_atom_types,
        cart_coords=noise_cart_coords,
        frac_coords=noise_frac_coords,
    )
    return batch_noise


def _random_rotation_matrix(validate: bool = False, **tensor_kwargs) -> torch.Tensor:
    """https://github.com/facebookresearch/all-atom-diffusion-transformer/src/models/components/kabsch_utils.py.

    Generates a random (3,3) rotation matrix.

    Args:
        tensor_kwargs: Keyword arguments to pass to the tensor constructor. E.g. `device`, `dtype`.

    Returns:
        A tensor of shape (3, 3) representing the rotation matrix.
    """
    # Generate a random quaternion
    q = torch.rand(4, **tensor_kwargs)
    q /= torch.linalg.norm(q)

    # Compute the rotation matrix from the quaternion
    rot_mat = torch.tensor(
        [
            [
                1 - 2 * q[2] ** 2 - 2 * q[3] ** 2,
                2 * q[1] * q[2] - 2 * q[0] * q[3],
                2 * q[1] * q[3] + 2 * q[0] * q[2],
            ],
            [
                2 * q[1] * q[2] + 2 * q[0] * q[3],
                1 - 2 * q[1] ** 2 - 2 * q[3] ** 2,
                2 * q[2] * q[3] - 2 * q[0] * q[1],
            ],
            [
                2 * q[1] * q[3] - 2 * q[0] * q[2],
                2 * q[2] * q[3] + 2 * q[0] * q[1],
                1 - 2 * q[1] ** 2 - 2 * q[2] ** 2,
            ],
        ],
        **tensor_kwargs,
    )

    if validate:
        assert torch.allclose(
            rot_mat @ rot_mat.T,
            torch.eye(3, device=rot_mat.device),
            atol=1e-5,
            rtol=1e-5,
        ), "Not a rotation matrix."

    return rot_mat
