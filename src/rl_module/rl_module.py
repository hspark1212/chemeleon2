# type: ignore
"""Reinforcement Learning PyTorch Lightning module using GRPO."""

import math
from collections import defaultdict
from functools import partial

import torch
from lightning import LightningModule

from src.data.schema import CrystalBatch
from src.ldm_module.diffusion import create_diffusion
from src.ldm_module.ldm_module import LDMModule


class RLModule(LightningModule):
    """Reinforcement Learning module using GRPO for fine-tuning."""

    def __init__(
        self,
        ldm_ckpt_path: str,
        rl_configs: dict,
        reward_fn: torch.nn.Module,
        sampling_configs: dict,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["reward_fn"])

        # Set automatic optimization to False to handle optimization manually
        self.automatic_optimization = False

        # Initialize hyperparameters
        self.clip_ratio = rl_configs.clip_ratio
        self.kl_weight = rl_configs.kl_weight
        self.entropy_weight = rl_configs.entropy_weight
        self.num_group_samples = rl_configs.num_group_samples
        self.group_reward_norm = rl_configs.group_reward_norm
        self.num_inner_batch = rl_configs.num_inner_batch
        self.reward_fn = reward_fn
        self.sampling_configs = sampling_configs

        # Load pre-trained LDM (Freeze VAE and condition embedding)
        self.ldm = LDMModule.load_from_checkpoint(ldm_ckpt_path)
        print(f"Loaded LDM from {ldm_ckpt_path}")
        self.ldm.vae.eval()
        for param in self.ldm.vae.parameters():
            param.requires_grad = False
        self.use_cfg = self.ldm.use_cfg

        # Sampling Diffusion
        if sampling_configs.sampler == "ddim":
            timestep_respacing = "ddim" + str(sampling_configs.sampling_steps)
        else:
            timestep_respacing = str(sampling_configs.sampling_steps)
        new_diffusion_configs = self.ldm.hparams.diffusion_configs.copy()
        new_diffusion_configs.update(timestep_respacing=timestep_respacing)
        self.sampling_diffusion = create_diffusion(**new_diffusion_configs)

    @torch.no_grad()
    def rollout(self, batch: CrystalBatch) -> dict:
        batch_gen = self.ldm.sample(batch, **self.sampling_configs)
        if self.use_cfg:
            batch_gen.zs, _ = torch.chunk(batch_gen.zs, 2, dim=1)  # (T+1, B, N, L)
            batch_gen.means, _ = torch.chunk(batch_gen.means, 2, dim=1)  # (T, B, N, L)
            batch_gen.stds, _ = torch.chunk(batch_gen.stds, 2, dim=1)  # (T, B, N, L)

        log_probs = []
        for i in range(self.sampling_diffusion.num_timesteps):
            log_prob = _calculate_log_prob(
                batch_gen.zs[i + 1],
                batch_gen.means[i],
                batch_gen.stds[i],
                batch_gen.mask,
            )
            log_probs.append(log_prob)
        log_probs = torch.stack(log_probs, dim=0)
        trajectory = dict(
            zs=batch_gen.zs,
            means=batch_gen.means,
            stds=batch_gen.stds,
            log_probs=log_probs,
            mask=batch_gen.mask,
            y=batch_gen.y,
        )
        return batch_gen, trajectory

    def compute_rewards(
        self, batch_gen: CrystalBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_samples = batch_gen.num_graphs
        rewards = self.reward_fn(batch_gen, device=self.device)
        if self.group_reward_norm:
            group_rewards_norm = []
            for i in range(0, num_samples, self.num_group_samples):
                group_reward = rewards[i : i + self.num_group_samples]
                group_reward_norm = self.reward_fn.normalize(group_reward)
                group_rewards_norm.append(group_reward_norm)
            rewards_norm = torch.cat(group_rewards_norm, dim=0)
        else:
            rewards_norm = self.reward_fn.normalize(rewards)
        return rewards, rewards_norm

    def calculate_loss(
        self,
        zs: torch.Tensor,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sampler_step_fn = (
            partial(self.sampling_diffusion.ddim_sample, eta=self.sampling_configs.eta)
            if self.sampling_configs.sampler == "ddim"
            else self.sampling_diffusion.p_sample
        )
        indices = list(range(self.sampling_diffusion.num_timesteps))[::-1]

        if self.use_cfg:
            assert y is not None
            y = self.ldm.condition_module(y, training=False).detach()

        res = defaultdict(int)
        for i, t in enumerate(indices):
            z = zs[i]
            old_log_probs = log_probs[i].detach()

            if self.use_cfg:
                z = torch.cat([z, z], dim=0)  # (2*B, N, L)
                mask = torch.cat([mask, mask], dim=0)  # (2*B, N)

            model_kwargs = dict(
                mask=mask,
                y=y,
                **(
                    {"cfg_scale": self.sampling_configs.cfg_scale}
                    if self.use_cfg
                    else {}
                ),
            )

            t = torch.tensor([t] * z.shape[0], device=z.device)
            out = sampler_step_fn(
                model=(
                    self.ldm.denoiser.forward_with_cfg
                    if self.use_cfg
                    else self.ldm.denoiser.forward
                ),
                x=z,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )

            if self.use_cfg:
                out["mean"], _ = torch.chunk(out["mean"], 2, dim=0)
                out["std"], _ = torch.chunk(out["std"], 2, dim=0)
                mask, _ = torch.chunk(mask, 2, dim=0)

            current_log_probs = _calculate_log_prob(
                zs[i + 1], out["mean"], out["std"], mask
            )

            # Compute surrogate objective
            if (t == 0).all() and (out["std"] == 0).all():
                continue  # Skip the last step when std is zero (ddim sampling is deterministic at t=0)
            log_ratio = current_log_probs - old_log_probs
            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(
                ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
            )
            surrogate_objective = torch.min(
                ratio * advantages, clipped_ratio * advantages
            )

            # Compute approximate KL divergence (k3 estimator)  http://joschu.net/blog/kl-approx.html
            kl_div_log_ratio = old_log_probs - current_log_probs
            kl_div = kl_div_log_ratio.exp() - 1 - kl_div_log_ratio

            # Compute entropy
            entropy = -current_log_probs

            # Compute mean loss
            policy_loss = (
                -surrogate_objective
                + self.kl_weight * kl_div
                - self.entropy_weight * entropy
            )
            loss = policy_loss.mean()
            scaled_loss = loss / len(indices)

            # Backpropagate loss
            self.manual_backward(scaled_loss)

            # Collect results
            res["scaled_loss"] += scaled_loss.detach().item()
            res["loss"] += loss.detach().item()
            res["sorrogate_objective"] += surrogate_objective.mean().detach().item()
            res["kl_div"] += kl_div.mean().detach().item()
            res["entropy"] += entropy.mean().detach().item()
            res["log_ratio"] = log_ratio.mean().detach().item()
            res["ratio"] = ratio.mean().detach().item()
        return res

    def training_step(self, batch: CrystalBatch, batch_idx: int) -> None:
        # Create total batch (batch_size * num_group_samples)
        total_batch = batch.repeat(self.num_group_samples)

        # Rollout
        batch_gen, trajectory = self.rollout(total_batch)
        assert (
            self.sampling_diffusion.num_timesteps  # [T]
            == len(trajectory["zs"]) - 1  # [T+1, B, N, L]
            == len(trajectory["log_probs"])  # [T, B]
        )

        # Compute rewards
        rewards, advantages = self.compute_rewards(batch_gen)

        # Initialize optimizer
        opt = self.optimizers()
        sch = self.lr_schedulers()

        # Inner loop for mini-batch gradient updates
        zs_chunked = torch.chunk(trajectory["zs"], self.num_inner_batch, dim=1)
        mask_chunked = torch.chunk(trajectory["mask"], self.num_inner_batch, dim=0)
        log_probs_chunked = torch.chunk(
            trajectory["log_probs"], self.num_inner_batch, dim=1
        )
        advantages_chunked = torch.chunk(advantages, self.num_inner_batch, dim=0)
        y_chunked = None
        if self.use_cfg:
            y = total_batch.y
            assert y is not None
            n = math.ceil(total_batch.batch_size / self.num_inner_batch)
            y_chunked = [
                {k: v[i * n : (i + 1) * n] for k, v in y.items()}
                for i in range(self.num_inner_batch)
            ]

        total_res = defaultdict(int)
        for i in range(self.num_inner_batch):
            opt.zero_grad()
            res = self.calculate_loss(
                zs=zs_chunked[i],
                log_probs=log_probs_chunked[i],
                advantages=advantages_chunked[i],
                mask=mask_chunked[i],
                y=y_chunked[i] if self.use_cfg else None,
            )
            torch.nn.utils.clip_grad_norm_(self.ldm.denoiser.parameters(), max_norm=1.0)
            opt.step()

            for k, v in res.items():
                total_res[k] += v / self.num_inner_batch

        if sch is not None:
            sch.step()

        # Log metrics
        total_res.update(
            dict(reward=rewards.mean().item(), advantages=advantages.mean().item())
        )
        self._log_metrics(res=total_res, split="train", batch_size=batch.num_graphs)

    def validation_step(self, batch: CrystalBatch, batch_idx: int) -> dict:
        # Create total batch (batch_size * num_group_samples)
        total_batch = CrystalBatch.from_data_list(
            [
                d.clone()
                for d in batch.to_data_list()
                for _ in range(self.num_group_samples)
            ]
        )

        # Rollout
        batch_gen, _ = self.rollout(total_batch)

        # Compute rewards
        rewards, advantages = self.compute_rewards(batch_gen)

        self._log_metrics(
            dict(reward=rewards.mean().item(), advantages=advantages.mean().item()),
            split="val",
            batch_size=batch.num_graphs,
        )

    def state_dict(self, *args, **kwargs):
        """Save the checkpoint of the diffusion_module than reinforce module."""
        return self.ldm.state_dict(*args, **kwargs)

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint["hyper_parameters"] = self.ldm.hparams

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
        optimizer = self.hparams.optimizer(params=self.ldm.denoiser.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "name": "learning rate",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


def _broadcast_mask(mask, z):
    while len(mask.shape) < len(z.shape):
        mask = mask[..., None]
    return mask.expand_as(z)


def _calculate_log_prob(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    mask: torch.Tensor,
    reduce: str = "mean",
) -> torch.Tensor:
    log_prob = -0.5 * (torch.log(2 * torch.pi * std**2) + ((x - mean) / std) ** 2)

    if mask is not None:
        log_prob = log_prob * _broadcast_mask(mask, x)

    reduced_dim = tuple(range(1, x.ndim))
    if reduce == "sum":
        log_prob = log_prob.sum(dim=reduced_dim)
    elif reduce == "mean":
        if mask is not None:
            summed = log_prob.sum(dim=reduced_dim)
            counts = _broadcast_mask(mask, x).sum(dim=reduced_dim).clamp(min=1e-6)
            log_prob = summed / counts
        else:
            log_prob = log_prob.mean(dim=reduced_dim)
    elif reduce == "none":
        pass
    return log_prob
