# type: ignore
"""Variational Autoencoder PyTorch Lightning module."""

import numpy as np
import torch
import torch.nn.functional as F
from ase import Atoms
from lightning import LightningModule
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from src.data.data_augmentation import apply_augmentation, apply_noise
from src.data.dataset_util import lattice_params_to_matrix_torch
from src.data.schema import CrystalBatch
from src.utils.timeout import timeout


class VAEModule(LightningModule):
    """Variational Autoencoder module for crystal structure encoding."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        latent_dim: int,
        loss_weights: dict,
        augmentation: dict,
        noise: dict,
        atom_type_predict: bool,
        structure_matcher: StructureMatcher,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)  # save and access via self.hparams

        # Encoder and decoder
        self.encoder = encoder
        self.decoder = decoder

        # Latent dimension
        self.quant_conv = torch.nn.Linear(
            self.encoder.hidden_dim, 2 * latent_dim, bias=False
        )
        self.post_quant_conv = torch.nn.Linear(
            latent_dim, self.decoder.hidden_dim, bias=False
        )

        # Foundation Alignment (FA) loss
        if self.hparams.loss_weights["fa"] > 0:
            self.proj = torch.nn.Linear(self.hparams.latent_dim, 256)  # MACE feat dim

        # Structure matcher for evaluation
        self.sm = structure_matcher

    def encode(self, batch: CrystalBatch) -> dict:
        encoded = self.encoder(batch)
        encoded["moments"] = self.quant_conv(encoded["x"])
        encoded["posterior"] = DiagonalGaussianDistribution(encoded["moments"])
        return encoded

    def decode(self, encoded: dict) -> dict:
        encoded["x"] = self.post_quant_conv(encoded["x"])
        decoder_out = self.decoder(encoded)
        return decoder_out

    def forward(self, batch: CrystalBatch) -> tuple:
        encoded = self.encode(batch)
        z = encoded["posterior"].sample()
        encoded["x"] = z
        encoded["z"] = z
        decoder_out = self.decode(encoded)
        return decoder_out, encoded

    def calculate_loss(self, batch: CrystalBatch, training: bool = True) -> dict:
        # Data augmentation when training
        if training:
            with torch.no_grad():
                batch_aug = apply_augmentation(
                    batch,
                    translate=self.hparams.augmentation.translate,
                    rotate=self.hparams.augmentation.rotate,
                )

                batch_noise = apply_noise(
                    batch_aug,
                    ratio=self.hparams.noise.ratio,
                    corruption_scale=self.hparams.noise.corruption_scale,
                )

        # Pass through encoder and decoder
        input_batch = batch_noise if training else batch  # pylint: disable=E0606
        true_batch = batch_aug if training else batch
        decoder_out, encoded = self(input_batch)

        # 1. Reconstruction loss
        loss_atom_types = 0
        if self.hparams.atom_type_predict:
            loss_atom_types = F.cross_entropy(
                decoder_out["atom_types"], true_batch.atom_types
            )
        loss_lengths = F.mse_loss(decoder_out["lengths"], true_batch.lengths_scaled)
        loss_angles = F.mse_loss(decoder_out["angles"], true_batch.angles_radians)
        loss_frac_coords = F.mse_loss(
            decoder_out["frac_coords"], true_batch.frac_coords
        )

        # 2. KL divergence loss
        loss_kl = encoded["posterior"].kl().mean()

        # 3. Foundation Alignment loss (optional)
        fa_loss = 0
        if self.hparams.loss_weights["fa"] > 0:
            z = self.proj(encoded["z"])
            mace_features = batch.mace_features
            z_norm = F.normalize(z, dim=-1)
            mace_features_norm = F.normalize(mace_features, dim=-1)
            z_cos_sim = torch.einsum("ij,kj->ik", z_norm, z_norm)
            mace_cos_sim = torch.einsum(
                "ij,kj->ik", mace_features_norm, mace_features_norm
            )
            diff = torch.abs(z_cos_sim - mace_cos_sim)
            fa_loss_1 = F.relu(diff - 0.25).mean()
            fa_loss_2 = F.relu(1 - 0.5 - F.cosine_similarity(mace_features, z)).mean()
            fa_loss = fa_loss_1 + fa_loss_2

        # Total loss
        loss = (
            self.hparams.loss_weights["atom_types"] * loss_atom_types
            + self.hparams.loss_weights["lengths"] * loss_lengths
            + self.hparams.loss_weights["angles"] * loss_angles
            + self.hparams.loss_weights["frac_coords"] * loss_frac_coords
            + self.hparams.loss_weights["kl"] * loss_kl
            + self.hparams.loss_weights["fa"] * fa_loss
        )

        return {
            "total_loss": loss.mean(),
            "loss_atom_types": loss_atom_types,
            "loss_lengths": loss_lengths,
            "loss_angles": loss_angles,
            "loss_frac_coords": loss_frac_coords,
            "loss_kl": loss_kl,
            "fa_loss": fa_loss,
        }

    def training_step(self, batch: CrystalBatch, batch_idx: int) -> torch.Tensor:
        res = self.calculate_loss(batch, training=True)
        if (
            self.trainer.current_epoch > 0
            and (self.trainer.current_epoch + 1) % self.trainer.check_val_every_n_epoch
            == 0
        ):
            structure_matching = self._compute_structure_matching(batch)
            res["structure_matching"] = structure_matching
        self._log_metrics(res, "train", batch_size=batch.num_graphs)
        return res["total_loss"]

    def validation_step(self, batch: CrystalBatch, batch_idx: int) -> dict:
        res = self.calculate_loss(batch, training=False)
        structure_matching = self._compute_structure_matching(batch)
        res["structure_matching"] = structure_matching
        self._log_metrics(res, "val", batch_size=batch.num_graphs)
        return res

    def test_step(self, batch: CrystalBatch, batch_idx: int) -> dict:
        res = self.calculate_loss(batch, training=False)
        structure_matching = self._compute_structure_matching(batch)
        res["structure_matching"] = structure_matching
        self._log_metrics(res, "test", batch_size=batch.num_graphs)
        return res

    def configure_optimizers(self) -> dict[str, any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "name": "learning rate",
                    # "monitor": "val/structure_matching",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    @torch.no_grad()
    def _log_metrics(
        self,
        res: dict,
        split: str,
        batch_size: int | None = None,
    ) -> None:
        for k, v in res.items():
            self.log(
                f"{split}/{k}",
                v,
                batch_size=batch_size,
                on_step=True if split == "train" else False,
                sync_dist=True,
                prog_bar=True,
            )

    @torch.no_grad()
    def sample(
        self,
        batch: CrystalBatch,
        return_atoms: bool = False,
        return_structures: bool = False,
    ) -> list[Atoms | Structure]:
        # Random latent vector
        z = torch.randn((batch.num_nodes, self.hparams.latent_dim), device=self.device)
        encoded = {
            "x": z,
            "num_atoms": batch.num_atoms,
            "batch": batch.batch,
            "token_idx": batch.token_idx,
        }
        decoder_out = self.decode(encoded)

        # Reconstruct the batch
        batch_recon = self.reconstruct(decoder_out, batch)

        # Return reconstructed batch either as Atoms or Structure object
        if return_atoms:
            samples = batch_recon.to_atoms()
        elif return_structures:
            samples = batch_recon.to_structure()
        else:
            samples = batch_recon
        return samples

    def reconstruct(self, decoder_out: dict, batch: CrystalBatch):
        # Reconstructed batch
        batch_recon = batch.clone()
        _atom_types = (
            decoder_out["atom_types"].argmax(dim=-1)
            if self.hparams.atom_type_predict
            else batch.atom_types
        )
        _atom_types[_atom_types == 0] = 1  # Prevent 0 atom type
        _frac_coords = decoder_out["frac_coords"]
        _lengths_scaled = decoder_out["lengths"]
        _lengths = _lengths_scaled * batch.num_atoms[:, None] ** (1 / 3)
        _angles_radians = decoder_out["angles"]
        _angles = torch.rad2deg(_angles_radians)
        _lattices = lattice_params_to_matrix_torch(_lengths, _angles)
        _cart_coords = torch.einsum("bij,bi->bj", _lattices[batch.batch], _frac_coords)
        batch_recon.update(
            pos=_cart_coords,
            atom_types=_atom_types,
            frac_coords=_frac_coords,
            cart_coords=_cart_coords,
            lattices=_lattices,
            lengths=_lengths,
            lengths_scaled=_lengths_scaled,
            angles=_angles,
            angles_radians=_angles_radians,
        )
        return batch_recon

    @torch.no_grad()
    def _compute_structure_matching(self, batch: CrystalBatch) -> float:
        # Sample from the VAE
        decoder_out, _ = self(batch)
        batch_recon = self.reconstruct(decoder_out, batch)
        rec_structures = batch_recon.to_structure()
        # Structure matching
        origin_structures = batch.to_structure()
        structure_matching = 0
        for orig, rec in zip(origin_structures, rec_structures, strict=False):
            cond = np.linalg.cond(rec.lattice.matrix)
            if cond > 1e3:
                continue
            try:
                match = timeout(seconds=3, default=False, verbose=True)(self.sm.fit)(
                    orig, rec
                )
                if match:
                    structure_matching += 1
            except Exception as e:
                print(f"Error in structure matching: {e}")
                continue
        structure_matching /= len(origin_structures)
        return structure_matching


class DiagonalGaussianDistribution:
    """Diagonal Gaussian distribution with mean and logvar parameters.

    Adapted from: https://github.com/CompVis/latent-diffusion, with modifications for our tensors,
    which are of shape (N, d) instead of (B, H, W, d) for 2D images.
    """

    def __init__(self, parameters, deterministic=False) -> None:
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(
            parameters, 2, dim=-1
        )  # split along channel dim
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=1
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=1,
                )

    def mode(self):
        return self.mean

    def __repr__(self) -> str:
        return f"DiagonalGaussianDistribution(mean={self.mean}, logvar={self.logvar})"
