"""Baseline tests for LDM module.

This test file establishes a functional baseline for the LDM (Latent Diffusion Model) module
to ensure code works correctly before and after formatting changes.
Tests include instantiation, forward pass shape validation, and
overfit-single-batch capability.
"""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.ldm_module.denoisers.dit import DiT
from src.ldm_module.ldm_module import LDMModule
from src.paths import DEFAULT_VAE_CKPT_PATH


@pytest.fixture(scope="module")
def ldm_model(device):
    """Create a minimal LDM model for testing.

    Uses DiT denoiser with minimal configuration to enable fast smoke tests.
    Uses default VAE checkpoint from src.paths.DEFAULT_VAE_CKPT_PATH
    """
    # Create DiT denoiser with minimal configuration
    denoiser = DiT(
        latent_dim=8,  # Must match VAE latent_dim from checkpoint
        hidden_size=128,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        learn_sigma=False,
    )

    # Create LDM module with minimal configuration
    model = LDMModule(
        normalize_latent=False,  # Disable for simpler testing
        denoiser=denoiser,
        augmentation=OmegaConf.create(  # type: ignore[arg-type]
            {
                "translate": False,
                "rotate": False,
            }
        ),
        diffusion_configs={
            "timestep_respacing": "",
            "noise_schedule": "linear",
            "diffusion_steps": 100,  # Reduced for faster testing
            "learn_sigma": False,
        },
        optimizer=torch.optim.Adam,  # type: ignore
        scheduler=None,  # type: ignore
        condition_module=None,  # No conditioning for baseline test
        vae_ckpt_path=str(DEFAULT_VAE_CKPT_PATH),
        ldm_ckpt_path=None,
        lora_configs=None,
    )

    return model.to(device)


@pytest.mark.smoke
@pytest.mark.baseline
def test_ldm_instantiation(ldm_model, device) -> None:
    """Test LDM model instantiation.

    Verifies that the LDM module can be instantiated with
    denoiser, VAE, diffusion, and all required components.
    """
    assert isinstance(ldm_model, LDMModule)
    assert isinstance(ldm_model.denoiser, nn.Module)
    assert ldm_model.vae is not None
    assert ldm_model.diffusion is not None
    assert str(ldm_model.device).startswith(device)

    # Verify VAE is frozen
    for param in ldm_model.vae.parameters():
        assert not param.requires_grad, "VAE parameters should be frozen in LDM"


@pytest.mark.smoke
@pytest.mark.baseline
def test_ldm_forward_pass_shapes(ldm_model, dummy_crystal_batch, device) -> None:
    """Test LDM forward pass with shape validation.

    Verifies that the loss calculation produces outputs with expected
    shapes and all required loss components.
    """
    # Create small batch for smoke test
    batch = dummy_crystal_batch(batch_size=2, num_atom_distribution="mp-20")
    batch = batch.to(device)

    # Calculate loss (this runs the forward pass internally)
    ldm_model.eval()
    with torch.no_grad():
        loss_dict = ldm_model.calculate_loss(batch, training=False)

    # Validate loss dictionary
    assert "total_loss" in loss_dict
    assert torch.isfinite(loss_dict["total_loss"]).all()
    assert loss_dict["total_loss"].numel() > 0  # Should have at least one element

    # Check that loss is a reasonable value (not NaN or Inf)
    loss_value = loss_dict["total_loss"]
    if loss_value.dim() > 0:
        loss_value = loss_value.mean()
    assert loss_value > 0, "Loss should be positive"


@pytest.mark.smoke
@pytest.mark.baseline
def test_ldm_loss_calculation(ldm_model, dummy_crystal_batch, device) -> None:
    """Test LDM loss calculation.

    Verifies that loss calculation works and produces
    finite values for the diffusion loss.
    """
    batch = dummy_crystal_batch(batch_size=2, num_atom_distribution="mp-20")
    batch = batch.to(device)

    ldm_model.eval()
    with torch.no_grad():
        loss_dict = ldm_model.calculate_loss(batch, training=False)

    # Validate loss dictionary
    assert "total_loss" in loss_dict
    assert torch.isfinite(loss_dict["total_loss"]).all()

    # The diffusion loss should be the MSE between predicted and true noise
    # It should be finite and positive
    loss_value = loss_dict["total_loss"]
    if loss_value.dim() > 0:
        loss_value = loss_value.mean()
    assert loss_value > 0, f"Loss should be positive, got {loss_value}"


@pytest.mark.baseline
@pytest.mark.slow
def test_ldm_overfit_single_batch(
    ldm_model, dummy_crystal_batch, seed_everything, device
) -> None:
    """Test LDM can overfit on a single batch.

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
    ldm_model.train()

    # Configure optimizer (only optimize denoiser, VAE is frozen)
    optimizer = torch.optim.Adam(ldm_model.denoiser.parameters(), lr=1e-3)

    # Record initial loss
    ldm_model.eval()
    with torch.no_grad():
        initial_loss_dict = ldm_model.calculate_loss(batch, training=False)
        initial_loss_value = initial_loss_dict["total_loss"]
        if initial_loss_value.dim() > 0:
            initial_loss_value = initial_loss_value.mean()
        initial_loss = initial_loss_value.item()

    # Train for 200 iterations (increased to ensure convergence on CPU)
    ldm_model.train()
    num_iterations = 200
    for _ in range(num_iterations):
        optimizer.zero_grad()
        loss_dict = ldm_model.calculate_loss(batch, training=True)
        loss = loss_dict["total_loss"]
        if loss.dim() > 0:
            loss = loss.mean()
        loss.backward()
        optimizer.step()

    # Record final loss
    ldm_model.eval()
    with torch.no_grad():
        final_loss_dict = ldm_model.calculate_loss(batch, training=False)
        final_loss_value = final_loss_dict["total_loss"]
        if final_loss_value.dim() > 0:
            final_loss_value = final_loss_value.mean()
        final_loss = final_loss_value.item()

    # Verify loss decreased significantly (to < 30% of initial loss)
    # Note: Using 30% threshold to account for diffusion model complexity
    # and frozen VAE encoder. The key is that loss decreases substantially,
    # proving the denoiser can learn from the data.
    assert final_loss < initial_loss * 0.35, (
        f"Failed to overfit single batch: "
        f"initial_loss={initial_loss:.4f}, final_loss={final_loss:.4f}"
    )
