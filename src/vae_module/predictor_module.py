# type: ignore
"""Property predictor PyTorch Lightning module."""

import torch
import torch.nn as nn
from lightning import LightningModule

from src.data.schema import CrystalBatch
from src.utils.scatter import scatter_mean, scatter_sum
from src.vae_module.vae_module import VAEModule


class PredictorModule(LightningModule):
    """Property predictor module for crystal structures."""

    def __init__(
        self,
        vae: VAEModule,
        target_conditions: dict,
        reduce: str,
        use_encoder_features: bool,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.vae = vae
        self.target_conditions = target_conditions
        self.reduce_fn = scatter_mean if reduce == "mean" else scatter_sum
        self.use_encoder_features = use_encoder_features

        # Projection layer
        self.input_dim = self.vae.hparams.latent_dim
        if use_encoder_features:
            self.input_dim += self.vae.encoder.hidden_dim
        self.num_targets = len(target_conditions)
        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 4),
            nn.LayerNorm(self.input_dim // 4),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(self.input_dim // 4, self.input_dim // 2),
            nn.LayerNorm(self.input_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(self.input_dim // 2, self.num_targets),
        )

        # Freeze VAE
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

    def forward(self, batch: CrystalBatch) -> torch.Tensor:
        # Get latent vector from VAE encoder
        with torch.no_grad():
            encoded = self.vae.encode(batch)
            x = encoded["posterior"].mode()  # (B_n, L)
            if self.use_encoder_features:
                x = torch.cat([x, encoded["x"]], dim=-1)  # (B_n, L + H)
            x = x.to(self.device)
        x = self.reduce_fn(x, batch.batch, dim=0)  # (B, L) or (B, L + H)
        x = self.proj(x)  # (B, num_targets)
        return x

    def normalize(self, x: torch.Tensor, condition: str) -> torch.Tensor:
        config = self.target_conditions[condition]
        if "mean" in config and "std" in config:
            mean = config["mean"]
            std = config["std"]
            return (x - mean) / (std + 1e-5)
        return x

    def denormalize(self, x: torch.Tensor, condition: str) -> torch.Tensor:
        config = self.target_conditions[condition]
        if "mean" in config and "std" in config:
            mean = config["mean"]
            std = config["std"]
            return x * (std + 1e-5) + mean
        return x

    def calculate_loss(self, batch: CrystalBatch) -> dict:
        # Check that batch contains all target conditions
        target_conditions = list(batch.y.keys())
        assert set(target_conditions) == set(self.target_conditions), (
            f"Expected conditions {self.target_conditions}, but got {target_conditions}"
        )

        # Forward pass
        pred = self.forward(batch)  # (B, num_targets)

        # Normalize
        targets = []
        for condition in self.target_conditions:
            target = batch.y[condition]  # (B,)
            target = self.normalize(target, condition)
            targets.append(target)

        targets = torch.stack(targets, dim=1).to(pred)  # (B, num_targets)

        # Calculate MSE loss
        loss = nn.functional.mse_loss(pred, targets)
        loss_dict = dict(loss=loss)

        # Calculate per-target losses for logging
        for i, condition in enumerate(self.target_conditions):
            loss_dict[f"loss_{condition}"] = nn.functional.mse_loss(
                pred[:, i], targets[:, i]
            )
            pred_denorm = self.denormalize(pred[:, i], condition)
            target_denorm = self.denormalize(targets[:, i], condition)
            loss_dict[f"loss_denorm_{condition}"] = nn.functional.mse_loss(
                pred_denorm, target_denorm
            )

        return loss_dict

    def training_step(self, batch: CrystalBatch, batch_idx: int):
        res = self.calculate_loss(batch)
        self._log_metrics(res, "train", batch_size=batch.num_graphs)
        return res["loss"]

    def validation_step(self, batch: CrystalBatch, batch_idx: int):
        res = self.calculate_loss(batch)
        self._log_metrics(res, "val", batch_size=batch.num_graphs)
        return res["loss"]

    def test_step(self, batch: CrystalBatch, batch_idx: int):
        res = self.calculate_loss(batch)
        self._log_metrics(res, "test", batch_size=batch.num_graphs)
        return res["loss"]

    def predict(self, batch: CrystalBatch) -> dict:
        self.eval()
        with torch.no_grad():
            pred = self.forward(batch)
            pred_dict = {}
            for i, condition in enumerate(self.target_conditions):
                pred_dict[condition] = self.denormalize(pred[:, i], condition)
        return pred_dict

    @torch.no_grad()
    def _log_metrics(
        self,
        res: dict,
        split: str,
        batch_size: int | None = None,
    ) -> None:
        for k, v in res.items():
            if isinstance(v, torch.Tensor):
                v = v.mean()
            self.log(
                f"{split}/{k}",
                v,
                batch_size=batch_size,
                on_step=True if split == "train" else False,
                sync_dist=True,
                prog_bar=True,
            )

    def configure_optimizers(self) -> dict[str, any]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "name": "learning rate",
                    "interval": "epoch",
                    "frequency": 10,
                },
            }
        return {"optimizer": optimizer}
