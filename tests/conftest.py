"""Shared pytest fixtures for chemeleon2 test suite.

This module provides common fixtures for baseline, contract, integration,
and unit tests. Fixtures include device detection, dummy crystal data,
and reproducibility helpers for PyTorch Lightning models.
"""

import pytest
import torch
import numpy as np


@pytest.fixture(scope="session")
def device():
    """Detect and return the available compute device (cuda/cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="function")
def seed_everything():
    """Set random seeds for reproducibility across numpy, torch, and Python."""

    def _seed(seed_value=42):
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

    This fixture uses the existing create_empty_batch utility to generate
    synthetic crystal structure data compatible with VAE, LDM, and RL modules.
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
        from src.data.schema import create_empty_batch
        from src.data.num_atom_distributions import NUM_ATOM_DISTRIBUTIONS

        distribution = NUM_ATOM_DISTRIBUTIONS[num_atom_distribution]
        num_atoms = np.random.choice(
            list(distribution.keys()),
            p=list(distribution.values()),
            size=batch_size,
        ).tolist()

        return create_empty_batch(num_atoms=num_atoms, device=device)

    return _create_batch
