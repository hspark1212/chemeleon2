# type: ignore
"""Latent Diffusion Model PyTorch Lightning module."""

from collections import defaultdict
from functools import partial

import torch
from lightning import LightningModule
from torch_geometric.utils import to_dense_batch

from src.data.data_augmentation import apply_augmentation
from src.data.schema import CrystalBatch
from src.ldm_module.condition import ConditionModule
from src.ldm_module.denoisers.dit import DiT
from src.ldm_module.diffusion import create_diffusion
from src.vae_module.vae_module import VAEModule


class LDMModule(LightningModule):
    """Latent Diffusion Model PyTorch Lightning module."""

    def __init__(
        self,
        normalize_latent: bool,
        denoiser: DiT,
        augmentation: dict | object,
        diffusion_configs: dict,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        condition_module: dict | None = None,
        vae_ckpt_path: str | None = None,
        ldm_ckpt_path: str | None = None,
        lora_configs: dict | None = None,
    ) -> None:
        super().__init__()

        # Normalize latent vectors
        self.normalize_latent = normalize_latent
        self.register_buffer("latent_std", torch.tensor(1.0))

        # Denoiser
        self.denoiser = denoiser

        # Diffusion
        self.diffusion = create_diffusion(**diffusion_configs)

        # Load pre-trained checkpoints
        if vae_ckpt_path is not None:
            self.vae = VAEModule.load_from_checkpoint(vae_ckpt_path, weights_only=False)
            print(f"Loaded VAE from {vae_ckpt_path}")
        if ldm_ckpt_path is not None:  # for finetuning
            ckpt = torch.load(ldm_ckpt_path, weights_only=False)
            if vae_ckpt_path is None:
                vae_ckpt_path = ckpt["hyper_parameters"]["vae_ckpt_path"]
                self.vae = VAEModule.load_from_checkpoint(
                    vae_ckpt_path, weights_only=False
                )
                print(f"Loaded VAE from {vae_ckpt_path} as in the LDM checkpoint.")
            self.load_state_dict(ckpt["state_dict"])
            print(f"Loaded pre-trained weights from {ldm_ckpt_path}")
        if ldm_ckpt_path is None and vae_ckpt_path is None:
            raise ValueError(
                "Either 'ldm_ckpt_path' or 'vae_ckpt_path' must be provided."
            )

        # Freeze VAE
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        # Condition
        self.use_cfg = False
        if condition_module is not None:
            self.use_cfg = True
            self.condition_module = ConditionModule(**condition_module)

        #  Lora
        if lora_configs is not None:
            from peft import LoraConfig, get_peft_model

            self.lora_configs = LoraConfig(**lora_configs)
            self.denoiser = get_peft_model(self.denoiser, self.lora_configs)
            self.denoiser.print_trainable_parameters()

        self.save_hyperparameters(logger=False, ignore=["ldm_ckpt_path"])

    @torch.no_grad()
    def _compute_latent_std(self, batch: CrystalBatch):
        x = self.vae.encode(batch)["posterior"].sample()
        return x.std()

    def on_train_batch_start(self, batch, batch_idx):
        if (
            self.normalize_latent
            and self.trainer.global_step == 0
            and self.latent_std == 1.0
        ):
            # Rank-0 computes, then broadcast for DDP
            if self.trainer.is_global_zero:
                std = self._compute_latent_std(batch)
                self.latent_std.copy_(std)
                print(f"Computed latent std: {self.latent_std.item()}")
            # Make all ranks share the same filled buffer
            self.trainer.strategy.broadcast(self.latent_std, src=0)
        return super().on_train_batch_start(batch, batch_idx)

    def calculate_loss(self, batch: CrystalBatch, training: bool = True) -> dict:
        if training:
            with torch.no_grad():
                batch = apply_augmentation(
                    batch,
                    translate=self.hparams.augmentation.translate,
                    rotate=self.hparams.augmentation.rotate,
                )

        with torch.no_grad():
            encoded = self.vae.encode(batch)
            x = encoded["posterior"].sample() / self.latent_std  # (B_n, L)
            x, mask = to_dense_batch(x, encoded["batch"])  # (B, N, L), (B, N)

        t = torch.randint(
            0, self.diffusion.num_timesteps, (x.shape[0],), device=self.device
        )
        y = None
        if self.use_cfg:
            y = batch.get("y")
            assert y is not None, "Batch must contain 'cond' key when use_cfg=True"
            y = self.condition_module(y, training=True)

        model_kwargs = dict(mask=mask, y=y)
        loss_dict = self.diffusion.training_losses(
            model=self.denoiser,
            x_start=x,
            t=t,
            model_kwargs=model_kwargs,
        )
        loss_dict["total_loss"] = loss_dict.pop("loss")
        return loss_dict

    def training_step(self, batch: CrystalBatch, batch_idx: int) -> torch.Tensor:
        res = self.calculate_loss(batch, training=True)
        self._log_metrics(res, "train", batch_size=batch.num_graphs)
        return res["total_loss"].mean()

    def validation_step(self, batch: CrystalBatch, batch_idx: int) -> dict:
        res = self.calculate_loss(batch, training=False)
        self._log_metrics(res, "val", batch_size=batch.num_graphs)
        return res

    def test_step(self, batch: CrystalBatch, batch_idx: int) -> dict:
        res = self.calculate_loss(batch, training=False)
        self._log_metrics(res, "test", batch_size=batch.num_graphs)
        return res

    @torch.no_grad()
    def sample(
        self,
        batch: CrystalBatch,
        sampler: str = "ddim",  # or "ddim"
        sampling_steps: int = 50,  # <= diffusion_configs["diffusion_steps"]
        eta: float = 1.0,  # Only used if sampler="ddim"
        cfg_scale: float = 2.0,  # Only used if use_cfg=True
        return_atoms: bool = False,
        return_structure: bool = False,
        collect_trajectory: bool = False,
        return_trajectory: bool = False,
        progress: bool = True,
    ):
        # Set up the sampler
        if sampler == "ddim":
            timestep_respacing = "ddim" + str(sampling_steps)
        else:
            timestep_respacing = str(sampling_steps)
        sampling_configs = self.hparams.diffusion_configs.copy()
        sampling_configs.update(timestep_respacing=timestep_respacing)
        sampling_diffusion = create_diffusion(**sampling_configs)
        sampler_fn = (
            partial(sampling_diffusion.ddim_sample_loop_progressive, eta=eta)
            if sampler == "ddim"
            else sampling_diffusion.p_sample_loop_progressive
        )
        if progress:
            print(
                f"Using {sampler} sampler_fn with {sampling_diffusion.num_timesteps} timesteps."
            )

        # Create random latent vectors
        z = torch.randn(
            (batch.num_nodes, self.vae.hparams.latent_dim), device=self.device
        )  # (B_n, L)
        z, mask = to_dense_batch(z, batch.batch)  # (B, N, L), (B, N)
        y = None
        if self.use_cfg:
            y = batch.get("y")
            assert y is not None, "Batch must contain 'y' key when use_cfg=True"
            z = torch.cat([z, z], dim=0)  # (2*B, N, L)
            mask = torch.cat([mask, mask], dim=0)  # (2*B, N)
            y = self.condition_module(y, training=False)  # (2*B, L_y)

        model_kwargs = dict(
            mask=mask,
            y=y,
            **({"cfg_scale": cfg_scale} if self.use_cfg else {}),
        )

        # Create a trajectory dictionary to collect samples
        trajectory = defaultdict(list)
        trajectory["z"].append(z)  # (B, N, L)

        # Function to decode latent vectors to structures
        def _decode_latent(diffusion_out, mask, batch):
            if self.use_cfg:
                diffusion_out, _ = diffusion_out.chunk(2, dim=0)  # (B, N, L)
                mask, _ = mask.chunk(2, dim=0)  # (B, N)
            diffusion_out = diffusion_out * self.latent_std

            # Decode the sample
            encoded_batch = dict(
                x=diffusion_out[mask],
                num_atoms=batch.num_atoms,
                batch=batch.batch,
                token_idx=batch.token_idx,
            )
            decoder_out = self.vae.decode(encoded_batch)
            return self.vae.reconstruct(decoder_out, batch)

        # Sample from the diffusion model
        for out in sampler_fn(
            model=(
                self.denoiser.forward_with_cfg
                if self.use_cfg
                else self.denoiser.forward
            ),
            shape=z.shape,
            noise=z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=progress,
            device=self.device,
        ):
            diffusion_out = out["sample"]
            if collect_trajectory:
                trajectory["z"].append(diffusion_out)  # (B, N, L)
                trajectory["mean"].append(out["mean"])  # (B, N, L)
                trajectory["std"].append(out["std"])  # (B, N, L)

            if return_trajectory:
                batch_rec = _decode_latent(diffusion_out, mask, batch)
                trajectory["atoms"].append(batch_rec.to_atoms())
                trajectory["structure"].append(batch_rec.to_structure())

        batch_rec = _decode_latent(diffusion_out, mask, batch)

        # Collect trajectory
        if collect_trajectory:
            for k, v in trajectory.items():
                setattr(batch_rec, f"{k}s", torch.stack(v, dim=0))
        batch_rec.mask = mask if not self.use_cfg else mask.chunk(2, dim=0)[0]

        # Return results
        if return_trajectory:
            return trajectory
        if return_atoms:
            return batch_rec.to_atoms()
        elif return_structure:
            return batch_rec.to_structure()
        return batch_rec

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
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "name": "learning rate",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
