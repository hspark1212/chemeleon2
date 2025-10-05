"""Utility functions for crystal structure dataset conversions.

This module provides conversion functions between pymatgen Structure objects,
PyTorch Geometric Data objects, and ASE Atoms objects.
"""

import torch
from ase import Atoms
from pymatgen.core import Lattice, Structure
from torch_geometric.data import Batch, Data


def pmg_structure_to_pyg_data(pmg_structure: Structure, **kwargs) -> Data:
    lengths = torch.as_tensor(
        pmg_structure.lattice.lengths, dtype=torch.float
    ).unsqueeze(0)
    angles = torch.as_tensor(pmg_structure.lattice.angles, dtype=torch.float).unsqueeze(
        0
    )
    return Data(
        pos=torch.as_tensor(pmg_structure.cart_coords, dtype=torch.float),
        atom_types=torch.as_tensor(pmg_structure.atomic_numbers, dtype=torch.long),
        frac_coords=torch.as_tensor(pmg_structure.frac_coords, dtype=torch.float),
        cart_coords=torch.as_tensor(pmg_structure.cart_coords, dtype=torch.float),
        lattices=torch.as_tensor(
            pmg_structure.lattice.matrix.copy(), dtype=torch.float
        ).unsqueeze(0),
        num_atoms=torch.as_tensor([len(pmg_structure)], dtype=torch.long),
        lengths=lengths,
        lengths_scaled=lengths / len(pmg_structure) ** (1 / 3),
        angles=angles,
        angles_radians=torch.deg2rad(angles),
        token_idx=torch.arange(len(pmg_structure), dtype=torch.long),
        **kwargs,
    )


def batch_to_atoms_list(batch: Batch, frac_coords: bool = True) -> list[Atoms]:
    atoms_list = []
    for data in batch.to_data_list():
        atoms = Atoms(
            numbers=data.atom_types.detach().cpu().numpy(),
            cell=data.lattices.squeeze(0).detach().cpu().numpy(),
            pbc=True,
        )
        if frac_coords:
            positions = data.frac_coords.detach().cpu().numpy()
            atoms.set_scaled_positions(positions)
        else:
            positions = data.cart_coords.detach().cpu().numpy()
            atoms.set_positions(positions)
        atoms_list.append(atoms)
    return atoms_list


def batch_to_structure_list(batch: Batch, frac_coords: bool = True) -> list[Structure]:
    structure_list = []
    for data in batch.to_data_list():
        # Get atomic numbers (convert to list for pymatgen Structure)
        atomic_numbers = data.atom_types.detach().cpu().numpy().tolist()

        # Create lattice from lattice matrix
        lattice = Lattice(data.lattices.squeeze(0).detach().cpu().numpy())

        if frac_coords:
            coords = data.frac_coords.detach().cpu().numpy()
            structure = Structure(
                lattice=Lattice.from_parameters(*lattice.parameters),
                species=atomic_numbers,
                coords=coords,
                coords_are_cartesian=False,
            )
        else:
            coords = data.cart_coords.detach().cpu().numpy()
            structure = Structure(
                lattice=Lattice.from_parameters(*lattice.parameters),
                species=atomic_numbers,
                coords=coords,
                coords_are_cartesian=True,
            )
        structure_list.append(structure)
    return structure_list


def lattice_params_to_matrix_torch(lengths, angles):
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1.0, 1.0)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack(
        [
            lengths[:, 0] * sins[:, 1],
            torch.zeros(lengths.size(0), device=lengths.device),
            lengths[:, 0] * coses[:, 1],
        ],
        dim=1,
    )
    vector_b = torch.stack(
        [
            -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
            lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
            lengths[:, 1] * coses[:, 0],
        ],
        dim=1,
    )
    vector_c = torch.stack(
        [
            torch.zeros(lengths.size(0), device=lengths.device),
            torch.zeros(lengths.size(0), device=lengths.device),
            lengths[:, 2],
        ],
        dim=1,
    )

    return torch.stack([vector_a, vector_b, vector_c], dim=1)
