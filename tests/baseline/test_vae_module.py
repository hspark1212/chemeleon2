"""Baseline tests for VAE module.

This test file establishes a functional baseline for the VAE module
to ensure code works correctly before and after formatting changes.
Tests include instantiation, forward pass shape validation, and
overfit-single-batch capability.
"""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from pymatgen.analysis.structure_matcher import StructureMatcher

from src.vae_module.decoders.transformer import TransformerDecoder
from src.vae_module.encoders.transformer import TransformerEncoder
from src.vae_module.vae_module import VAEModule


@pytest.fixture(scope="module")
def vae_model(device):
    """Create a minimal VAE model for testing.

    Uses TransformerEncoder and TransformerDecoder with minimal
    configuration to enable fast smoke tests.
    """
    # Create encoder
    encoder = TransformerEncoder(
        max_num_elements=100,
        d_model=128,
        nhead=4,
        dim_feedforward=256,
        dropout=0.0,
        num_layers=2,
    )

    # Create decoder
    decoder = TransformerDecoder(
        max_num_elements=100,
        d_model=128,
        nhead=4,
        dim_feedforward=256,
        dropout=0.0,
        num_layers=2,
        atom_type_predict=True,
    )

    # Create VAE module with minimal configuration
    # Use OmegaConf to support dot notation in hparams
    model = VAEModule(
        encoder=encoder,
        decoder=decoder,
        latent_dim=64,
        loss_weights=OmegaConf.create(  # type: ignore
            {
                "atom_types": 1.0,
                "lengths": 1.0,
                "angles": 1.0,
                "frac_coords": 1.0,
                "kl": 0.01,
                "fa": 0.0,
            }
        ),
        augmentation=OmegaConf.create(  # type: ignore
            {
                "translate": False,
                "rotate": False,
            }
        ),
        noise=OmegaConf.create(  # type: ignore
            {
                "ratio": 0.0,
                "corruption_scale": 0.0,
            }
        ),
        atom_type_predict=True,
        structure_matcher=StructureMatcher(),
        optimizer=torch.optim.Adam,  # type: ignore
        scheduler=None,  # type: ignore
    )

    return model.to(device)


@pytest.mark.smoke
@pytest.mark.baseline
def test_vae_instantiation(vae_model, device) -> None:
    """Test VAE model instantiation.

    Verifies that the VAE module can be instantiated with
    encoder, decoder, and all required components.
    """
    assert isinstance(vae_model, VAEModule)
    assert isinstance(vae_model.encoder, nn.Module)
    assert isinstance(vae_model.decoder, nn.Module)
    assert vae_model.quant_conv is not None
    assert vae_model.post_quant_conv is not None
    assert str(vae_model.device).startswith(device)


@pytest.mark.smoke
@pytest.mark.baseline
def test_vae_forward_pass_shapes(vae_model, dummy_crystal_batch, device) -> None:
    """Test VAE forward pass with shape validation.

    Verifies that the forward pass produces outputs with expected
    shapes matching the input batch dimensions.
    """
    # Create small batch for smoke test
    batch = dummy_crystal_batch(batch_size=2, num_atom_distribution="mp-20")
    batch = batch.to(device)

    # Forward pass
    vae_model.eval()
    with torch.no_grad():
        decoder_out, encoded = vae_model(batch)

    # Validate encoded outputs
    assert "posterior" in encoded
    assert "z" in encoded
    assert encoded["z"].shape[0] == batch.num_nodes
    assert encoded["z"].shape[1] == vae_model.hparams.latent_dim

    # Validate decoder outputs
    assert "atom_types" in decoder_out
    assert "lengths" in decoder_out
    assert "angles" in decoder_out
    assert "frac_coords" in decoder_out

    # Shape validation
    assert decoder_out["atom_types"].shape[0] == batch.num_nodes
    assert decoder_out["frac_coords"].shape == (batch.num_nodes, 3)
    assert decoder_out["lengths"].shape[0] == batch.num_graphs
    assert decoder_out["angles"].shape[0] == batch.num_graphs


@pytest.mark.smoke
@pytest.mark.baseline
def test_vae_loss_calculation(vae_model, dummy_crystal_batch, device) -> None:
    """Test VAE loss calculation.

    Verifies that loss calculation works and produces
    finite values for all loss components.
    """
    batch = dummy_crystal_batch(batch_size=2, num_atom_distribution="mp-20")
    batch = batch.to(device)

    vae_model.eval()
    with torch.no_grad():
        loss_dict = vae_model.calculate_loss(batch, training=False)

    # Validate loss dictionary
    assert "total_loss" in loss_dict
    assert torch.isfinite(loss_dict["total_loss"])
    assert loss_dict["total_loss"] > 0

    # Validate loss components
    expected_keys = [
        "loss_atom_types",
        "loss_lengths",
        "loss_angles",
        "loss_frac_coords",
        "loss_kl",
    ]
    for key in expected_keys:
        assert key in loss_dict
        assert torch.isfinite(loss_dict[key])


@pytest.mark.baseline
@pytest.mark.slow
def test_vae_overfit_single_batch(
    vae_model, dummy_crystal_batch, seed_everything, device
) -> None:
    """Test VAE can overfit on a single batch.

    Critical validation test following Karpathy's principle:
    'If you can't overfit on a tiny batch, things are definitely broken.'

    Trains on a single batch for 200 iterations and verifies
    that loss decreases significantly (to < 35% of initial loss).
    """
    seed_everything(42)

    # Create single batch
    batch = dummy_crystal_batch(batch_size=2, num_atom_distribution="mp-20")
    batch = batch.to(device)

    # Set model to training mode
    vae_model.train()

    # Configure optimizer
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-3)

    # Record initial loss
    with torch.no_grad():
        initial_loss_dict = vae_model.calculate_loss(batch, training=False)
        initial_loss = initial_loss_dict["total_loss"].item()

    # Train for 200 iterations
    num_iterations = 200
    for _ in range(num_iterations):
        optimizer.zero_grad()
        loss_dict = vae_model.calculate_loss(batch, training=True)
        loss = loss_dict["total_loss"]
        loss.backward()
        optimizer.step()

    # Record final loss
    vae_model.eval()
    with torch.no_grad():
        final_loss_dict = vae_model.calculate_loss(batch, training=False)
        final_loss = final_loss_dict["total_loss"].item()

    # Verify loss decreased significantly
    assert final_loss < initial_loss * 0.35, (
        f"Failed to overfit single batch: "
        f"initial_loss={initial_loss:.4f}, final_loss={final_loss:.4f}"
    )
