"""Shared pytest fixtures for chemeleon2 test suite.

This module provides common fixtures for baseline, contract, integration,
and unit tests. Fixtures include device detection, dummy crystal data,
and reproducibility helpers for PyTorch Lightning models.
"""

import numpy as np
import pytest
import torch


@pytest.fixture(scope="session")
def device() -> str:
    """Detect and return the available compute device (cuda/cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="function")
def seed_everything():
    """Set random seeds for reproducibility across numpy, torch, and Python."""

    def _seed(seed_value=42) -> None:
        import random

        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return _seed


@pytest.fixture(scope="function")
def dummy_crystal_batch(device):
    """Create a small dummy CrystalBatch for testing model forward passes.

    This fixture generates synthetic crystal structure data compatible with
    VAE, LDM, and RL modules. All tensors are initialized with valid values
    to avoid CUDA device assertions and NaN issues.
    Uses realistic atom count distributions from num_atom_distributions.
    """

    def _create_batch(batch_size=2, num_atom_distribution="mp-20"):
        """Generate dummy CrystalBatch with specified dimensions.

        Args:
            batch_size: Number of crystal structures in batch
            num_atom_distribution: Distribution name ("mp-20" or "mp-120")

        Returns:
            CrystalBatch object ready for model testing
        """
        from torch_geometric.data import Data

        from src.data.num_atom_distributions import NUM_ATOM_DISTRIBUTIONS
        from src.data.schema import CrystalBatch

        distribution = NUM_ATOM_DISTRIBUTIONS[num_atom_distribution]
        num_atoms = np.random.choice(
            list(distribution.keys()),
            p=list(distribution.values()),
            size=batch_size,
        ).tolist()

        # Generate properly initialized dummy data for each structure
        data_list = []
        for n in num_atoms:
            # Generate random atom types (1-99, avoiding 0)
            atom_types = torch.randint(1, 100, (n,), dtype=torch.long)

            # Generate random fractional coordinates (0-1)
            frac_coords = torch.rand((n, 3))

            # Generate random lattice (identity matrix with small perturbations)
            lattice = torch.eye(3).unsqueeze(0) + torch.randn((1, 3, 3)) * 0.1

            # Generate lattice parameters
            lengths = torch.rand((1, 3)) * 5 + 5  # Between 5-10 Angstroms
            lengths_scaled = lengths / (n ** (1 / 3))  # Scale by num_atoms^(1/3)
            angles = torch.ones((1, 3)) * (torch.pi / 2)  # 90 degrees
            angles_radians = angles.clone()

            # Calculate cartesian coordinates
            cart_coords = torch.einsum("bij,ni->nj", lattice, frac_coords)

            data_list.append(
                Data(
                    pos=cart_coords,
                    atom_types=atom_types,
                    frac_coords=frac_coords,
                    cart_coords=cart_coords,
                    lattices=lattice,
                    num_atoms=torch.as_tensor(n, dtype=torch.long),
                    lengths=lengths,
                    lengths_scaled=lengths_scaled,
                    angles=torch.rad2deg(angles),
                    angles_radians=angles_radians,
                    token_idx=torch.arange(n, dtype=torch.long),
                )
            )

        return CrystalBatch.from_data_list(data_list).to(device=device)

    return _create_batch
