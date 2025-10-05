"""Baseline tests for data loading.

This test file establishes a functional baseline for data loading
to ensure dataloaders work correctly before and after formatting changes.
Tests include dataloader batching with shape validation, label alignment,
and dtype verification.
"""

import pytest
import torch

from src.data.schema import CrystalBatch


@pytest.mark.smoke
@pytest.mark.baseline
def test_dataloader_batching(dummy_crystal_batch, device) -> None:
    """Test dataloader batching with shape validation.

    Verifies that CrystalBatch properly batches crystal structures
    with correct shapes for all tensor attributes.
    """
    # Create batch with multiple structures
    batch_size = 4
    batch = dummy_crystal_batch(batch_size=batch_size, num_atom_distribution="mp-20")

    # Verify batch is CrystalBatch
    assert isinstance(batch, CrystalBatch)

    # Verify batch contains expected number of structures
    assert batch.num_graphs == batch_size

    # Verify tensor shapes for graph-level attributes
    # Lattice parameters: one per structure
    assert batch.lengths.shape[0] == batch_size
    assert batch.lengths.shape[1] == 3  # a, b, c lengths
    assert batch.angles.shape[0] == batch_size
    assert batch.angles.shape[1] == 3  # alpha, beta, gamma angles
    assert batch.lattices.shape == (batch_size, 3, 3)  # Lattice matrix

    # Verify tensor shapes for node-level attributes
    # Total number of atoms across all structures
    total_atoms = batch.num_nodes
    assert batch.atom_types.shape[0] == total_atoms
    assert batch.frac_coords.shape == (total_atoms, 3)
    assert batch.cart_coords.shape == (total_atoms, 3)
    assert batch.pos.shape == (total_atoms, 3)

    # Verify token_idx for positional encoding
    assert batch.token_idx.shape[0] == total_atoms


@pytest.mark.smoke
@pytest.mark.baseline
def test_label_alignment(dummy_crystal_batch, device) -> None:
    """Test label alignment - count matches structure count.

    Verifies that graph-level labels (lattice parameters) match
    the number of structures in the batch, and node-level labels
    (atom properties) match the total number of atoms.
    """
    batch_size = 3
    batch = dummy_crystal_batch(batch_size=batch_size, num_atom_distribution="mp-20")

    # Graph-level label counts should match batch_size
    assert batch.lengths.shape[0] == batch_size, (
        f"Length labels count ({batch.lengths.shape[0]}) "
        f"does not match batch size ({batch_size})"
    )
    assert batch.angles.shape[0] == batch_size, (
        f"Angle labels count ({batch.angles.shape[0]}) "
        f"does not match batch size ({batch_size})"
    )
    assert batch.lattices.shape[0] == batch_size, (
        f"Lattice labels count ({batch.lattices.shape[0]}) "
        f"does not match batch size ({batch_size})"
    )

    # Node-level label counts should match total atom count
    total_atoms = batch.num_nodes
    assert batch.atom_types.shape[0] == total_atoms, (
        f"Atom type labels count ({batch.atom_types.shape[0]}) "
        f"does not match total atom count ({total_atoms})"
    )
    assert batch.frac_coords.shape[0] == total_atoms, (
        f"Fractional coordinate labels count ({batch.frac_coords.shape[0]}) "
        f"does not match total atom count ({total_atoms})"
    )

    # Verify batch pointer integrity
    # batch.batch maps each node to its graph index
    assert batch.batch.shape[0] == total_atoms
    assert batch.batch.min() == 0
    assert batch.batch.max() == batch_size - 1


@pytest.mark.smoke
@pytest.mark.baseline
def test_dtypes(dummy_crystal_batch, device) -> None:
    """Test dtypes - float32 for coordinates, long for atom types.

    Verifies that all tensors in CrystalBatch have the correct
    data types for downstream model compatibility.
    """
    batch = dummy_crystal_batch(batch_size=2, num_atom_distribution="mp-20")

    # Float tensors for continuous values
    float_attributes = [
        "frac_coords",
        "cart_coords",
        "pos",
        "lengths",
        "lengths_scaled",
        "angles",
        "angles_radians",
        "lattices",
    ]
    for attr in float_attributes:
        tensor = getattr(batch, attr)
        assert tensor.dtype == torch.float32, (
            f"{attr} has dtype {tensor.dtype}, expected torch.float32"
        )

    # Long/integer tensors for discrete values
    long_attributes = [
        "atom_types",
        "num_atoms",
        "token_idx",
        "batch",
    ]
    for attr in long_attributes:
        tensor = getattr(batch, attr)
        assert tensor.dtype == torch.long, (
            f"{attr} has dtype {tensor.dtype}, expected torch.long"
        )


@pytest.mark.smoke
@pytest.mark.baseline
def test_device_placement(dummy_crystal_batch, device) -> None:
    """Test that batch tensors are on the correct device.

    Verifies that all tensors in CrystalBatch are placed on
    the expected device (CPU or CUDA).
    """
    batch = dummy_crystal_batch(batch_size=2, num_atom_distribution="mp-20")

    # Check all tensor attributes are on the correct device
    tensor_attributes = [
        "atom_types",
        "frac_coords",
        "cart_coords",
        "pos",
        "lattices",
        "lengths",
        "angles",
        "num_atoms",
        "token_idx",
    ]

    for attr in tensor_attributes:
        tensor = getattr(batch, attr)
        assert str(tensor.device).startswith(device), (
            f"{attr} is on device {tensor.device}, expected {device}"
        )
