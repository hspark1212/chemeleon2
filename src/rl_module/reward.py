"""Reward functions for reinforcement learning."""

import torch

from src.data.schema import CrystalBatch
from src.rl_module.components import RewardComponent, normalize, standardize
from src.utils.metrics import Metrics


class ReinforceReward(torch.nn.Module):
    """Reward function for RL module."""

    def __init__(
        self,
        components: list[RewardComponent],
        normalize_fn: str | None = None,
        eps: float = 1e-4,
        reference_dataset: str = "mp-20",
    ):
        super().__init__()
        self.components = torch.nn.ModuleList(components)
        self.normalize_fn = normalize_fn
        self.eps = eps

        # Collect all required metrics from components
        all_metrics = set()
        for component in components:
            if hasattr(component, "required_metrics"):
                all_metrics.update(component.required_metrics)

        # Create single Metrics object if needed
        if all_metrics:
            self.metrics = Metrics(
                metrics=list(all_metrics),
                reference_dataset=reference_dataset,
                progress_bar=False,
            )
        else:
            self.metrics = None

    @torch.no_grad()
    def forward(
        self, batch_gen: CrystalBatch, device: torch.device | None = None
    ) -> torch.Tensor:
        gen_structures = batch_gen.to_structure()

        # Collect metrics if needed
        if self.metrics is not None:
            with torch.enable_grad():  # Enable grad for mace-torch
                self.metrics.compute(gen_structures=gen_structures)

        # Compute rewards from all components
        device = device if device is not None else batch_gen.pos.device
        total_rewards = torch.zeros(len(batch_gen), device=device)
        for component in self.components:
            comp_rewards = component(
                batch_gen=batch_gen,
                gen_structures=gen_structures,
                metrics_obj=self.metrics,
                device=device,
            )
            total_rewards += comp_rewards

        if torch.isnan(total_rewards).any():
            raise ValueError("NaN values found in rewards.")

        return total_rewards

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.normalize_fn == "norm":
            rewards = normalize(rewards, eps=self.eps)
        elif self.normalize_fn == "std":
            rewards = standardize(rewards, eps=self.eps)
        elif self.normalize_fn == "subtract_mean":
            rewards = rewards - rewards.mean()
        elif self.normalize_fn == "clip":
            rewards = torch.clamp(rewards, min=-1.0, max=1.0)
        elif self.normalize_fn is None:
            pass
        else:
            raise ValueError(
                f"Unknown normalization type: {self.normalize_fn}. Use 'norm', 'std', 'clip', or None."
            )
        return rewards
