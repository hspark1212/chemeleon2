"""Unit tests for structure featurizer."""

import pytest
import torch
from pymatgen.core import Lattice, Structure

from src.utils.featurizer import featurize


@pytest.fixture
def dummy_structures():
    """Create dummy pymatgen structures for testing."""
    lattice = Lattice.cubic(5.0)
    structures = [
        Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]]),
        Structure(lattice, ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    ]
    return structures


@pytest.mark.unit
def test_featurize_basic(dummy_structures):
    """Test basic featurization output structure."""
    result = featurize(dummy_structures, batch_size=2, device="cpu")

    # Check output keys
    assert "structure_features" in result
    assert "composition_features" in result
    assert "atom_features" in result


@pytest.mark.unit
def test_featurize_shapes(dummy_structures):
    """Test output tensor shapes."""
    result = featurize(dummy_structures, batch_size=2, device="cpu")

    # Structure features: (num_structures, feature_dim)
    assert result["structure_features"].shape[0] == len(dummy_structures)
    assert result["structure_features"].dim() == 2

    # Composition features: (num_structures, feature_dim)
    assert result["composition_features"].shape[0] == len(dummy_structures)
    assert result["composition_features"].dim() == 2

    # Atom features: list of tensors (one per structure)
    assert len(result["atom_features"]) == len(dummy_structures)
    assert all(isinstance(f, torch.Tensor) for f in result["atom_features"])


@pytest.mark.unit
def test_featurize_batch_processing(dummy_structures):
    """Test batch processing with different batch sizes."""
    # Small batch size
    result_small = featurize(dummy_structures, batch_size=1, device="cpu")

    # Large batch size
    result_large = featurize(dummy_structures, batch_size=10, device="cpu")

    # Results should be identical regardless of batch size
    assert torch.allclose(
        result_small["structure_features"],
        result_large["structure_features"],
        atol=1e-5,
    )


@pytest.mark.unit
def test_featurize_encoder_features(dummy_structures):
    """Test with encoder features included."""
    result_base = featurize(dummy_structures, use_encoder_features=False, device="cpu")
    result_with_encoder = featurize(
        dummy_structures, use_encoder_features=True, device="cpu"
    )

    # With encoder features, dimension should be larger
    assert (
        result_with_encoder["structure_features"].shape[1]
        > result_base["structure_features"].shape[1]
    )


@pytest.mark.unit
def test_featurize_reduce_methods(dummy_structures):
    """Test different reduction methods (mean vs sum)."""
    result_mean = featurize(dummy_structures, reduce="mean", device="cpu")
    result_sum = featurize(dummy_structures, reduce="sum", device="cpu")

    # Both should have same shape
    assert (
        result_mean["structure_features"].shape
        == result_sum["structure_features"].shape
    )

    # But different values
    assert not torch.allclose(
        result_mean["structure_features"], result_sum["structure_features"]
    )
