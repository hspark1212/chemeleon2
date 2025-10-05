"""Reward functions for reinforcement learning fine-tuning."""

import enum
from collections import defaultdict
from functools import partial

import amd
import numpy as np
import torch
from pymatgen.core import Composition

from src.data.schema import CrystalBatch
from src.utils.featurizer import featurize
from src.utils.metrics import Metrics, structures_to_amd
from src.vae_module.predictor_module import PredictorModule


class RewardType(enum.Enum):
    """Enum for different reward types."""

    DNG = "dng"
    CSP = "csp"
    CUSTOM = "custom"
    BANDGAP = "bandgap"


class ReinforceReward(torch.nn.Module):
    """Reward function for reinforcement learning."""

    def __init__(
        self,
        reward_type: str,
        normalize_fn: str,
        eps: float = 1e-4,
        reference_dataset: str = "mp-20",
        **kwargs,
    ) -> None:
        super().__init__()
        print(f"Starting setup for reward type: {reward_type}")
        self.reward_type = RewardType(reward_type)
        self.normalize_fn = normalize_fn
        if normalize_fn not in ["norm", "std", "subtract_mean", "clip", None]:
            raise ValueError(
                f"Unknown normalization function: {normalize_fn}. Use 'norm', 'std', 'subtract_mean', 'clip', or None."
            )
        self.eps = eps
        if self.reward_type == RewardType.DNG:
            m = Metrics(
                metrics=[
                    "unique",
                    "novel",
                    "composition_validity",
                    "e_above_hull",
                    "structure_diversity",
                    "composition_diversity",
                ],
                progress_bar=False,
                reference_dataset=reference_dataset,
            )
            reward_fn = partial(reward_dng, m=m, **kwargs)
        elif self.reward_type == RewardType.CSP:
            m = Metrics(
                metrics=[
                    "unique",
                    "e_above_hull",
                ],
                progress_bar=False,
                reference_dataset=reference_dataset,
            )
            reward_fn = partial(reward_csp, m=m, **kwargs)
        elif self.reward_type == RewardType.CUSTOM:
            reward_fn = custom_reward
        elif self.reward_type == RewardType.BANDGAP:
            if "predictor" not in kwargs:
                raise ValueError("Predictor model must be provided for bandgap reward.")
            predictor: PredictorModule = kwargs["predictor"]
            predictor.eval()
            reward_fn = partial(
                reward_bandgap,
                predictor=predictor,
                target_value=kwargs.get("target_bandgap"),
            )
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")
        self.reward_fn = reward_fn
        print("Reward function setup complete.")

    def forward(
        self, batch_gen: CrystalBatch, device: torch.device | None = None
    ) -> torch.Tensor:
        # Calculate reward
        rewards = self.reward_fn(batch_gen=batch_gen)
        if device is not None:
            rewards = rewards.to(device)

        # If any reward is NaN, raise an error
        if torch.isnan(rewards).any():
            raise ValueError(
                "NaN values found in rewards. Please check the reward function."
            )
        return rewards

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.normalize_fn == "norm":
            rewards = normalize(rewards, eps=self.eps)
        elif self.normalize_fn == "std":
            rewards = standardize(rewards, eps=self.eps)
        elif self.normalize_fn == "subtract_mean":
            # Normalize rewards just by subtracting the mean (Dr. GRPO)
            rewards = rewards - rewards.mean()
        elif self.normalize_fn == "clip":
            # Clip rewards to [-1, 1]
            rewards = torch.clamp(rewards, min=-1.0, max=1.0)
        elif self.normalize_fn is None:
            pass
        else:
            raise ValueError(
                f"Unknown normalization type: {self.normalize_fn}. Use 'norm', 'std', 'clip', or None."
            )

        return rewards


def reward_dng(batch_gen: CrystalBatch, m: Metrics, **kwargs) -> torch.Tensor:
    gen_structures = batch_gen.to_structure()
    with torch.enable_grad():
        results = m.compute(gen_structures=gen_structures)

    # 1. Unique and Novel Reward
    ref_structures = m._reference_structures
    ref_structures_by_formula = defaultdict(list)
    for ref_structure in ref_structures + gen_structures:
        ref_structures_by_formula[ref_structure.reduced_formula].append(ref_structure)
    rewards = []
    for i, gen_structure in enumerate(gen_structures):
        u, v = results["unique"][i], results["novel"][i]
        if u and v:
            r = 1
        elif not u and not v:
            r = 0
        else:
            matching_refs = ref_structures_by_formula.get(
                gen_structure.reduced_formula, []
            )
            amds = structures_to_amd([gen_structure] + matching_refs, 100)
            dists = amd.AMD_cdist(amds, amds)[0]
            r = dists[dists > 0].min()
        rewards.append(r)
    r_creativity = torch.as_tensor(rewards).float()

    # 2. Energy above hull Reward
    r_energy = torch.as_tensor(results["e_above_hull"]).float()
    r_energy = r_energy.nan_to_num(nan=1.0)  # max clip energy
    r_energy = r_energy.clamp(min=0.0, max=1.0)
    r_energy = r_energy * -1.0  # Negative for minimization
    r_energy = normalize(r_energy)

    # # 3. Structure diversity
    gen_features = featurize(gen_structures)
    gen_structure_features = gen_features["structure_features"]
    assert m._reference_structure_features is not None
    ref_structure_features = m._reference_structure_features.to(
        gen_structure_features.device
    )
    if len(ref_structure_features) > 50000:  # Limit to 50,000 to avoid memory issues
        indices = torch.randperm(len(ref_structure_features))[:50000]
        ref_structure_features = ref_structure_features[indices]
    r_structure_diversity = mmd_reward(
        z_gen=gen_structure_features, z_ref=ref_structure_features
    )["r_indiv"]
    r_structure_diversity = normalize(r_structure_diversity)

    # 4. Composition diversity
    gen_composition_features = gen_features["composition_features"]
    assert m._reference_composition_features is not None
    ref_composition_features = m._reference_composition_features.to(
        gen_composition_features.device
    )
    if len(ref_composition_features) > 50000:  # Limit to 50,000 to avoid memory issues
        indices = torch.randperm(len(ref_composition_features))[:50000]
        ref_composition_features = ref_composition_features[indices]
    r_composition_diversity = mmd_reward(
        z_gen=gen_composition_features, z_ref=ref_composition_features
    )["r_indiv"]
    r_composition_diversity = normalize(r_composition_diversity)

    # Total rewards
    total_rewards = (
        kwargs.get("weight_r_creativity", 1.0) * r_creativity
        + kwargs.get("weight_r_energy", 1.0) * r_energy
        + kwargs.get("weight_r_structure_diversity", 0.1) * r_structure_diversity
        + kwargs.get("weight_r_composition_diversity", 1.0) * r_composition_diversity
    )
    return total_rewards


def reward_csp(batch_gen: CrystalBatch, m: Metrics) -> torch.Tensor:
    gen_structures = batch_gen.to_structure()
    with torch.enable_grad():
        results = m.compute(gen_structures=gen_structures)
    # 1. Unique Reward
    amds = structures_to_amd(gen_structures, 100)
    dists_all = amd.AMD_cdist(amds, amds)
    np.fill_diagonal(dists_all, np.inf)  # Avoid self-distance
    min_dists = dists_all.min(axis=0)
    rewards = np.where(results["unique"], 1.0, min_dists)
    r_creativity = torch.as_tensor(rewards).float()

    # 2. Energy above hull Reward
    r_energy = torch.as_tensor(results["e_above_hull"]).float()
    r_energy = r_energy.nan_to_num(nan=1.0)  # max clip energy
    r_energy = r_energy.clamp(min=0.0, max=1.0)
    r_energy = r_energy * -1.0  # Negative for minimization
    r_energy = normalize(r_energy)

    # 3. Composition matching
    gen_compositions = [st.reduced_formula for st in gen_structures]
    y = getattr(batch_gen, "y", None)
    assert y is not None and "composition" in y
    ref_compositions = [Composition(comp).reduced_formula for comp in y["composition"]]
    r_composition_matching = torch.tensor(
        [
            gen_comp == ref_comp
            for gen_comp, ref_comp in zip(
                gen_compositions, ref_compositions, strict=False
            )
        ]
    )

    # Total rewards
    total_rewards = 1.0 * r_creativity + 1.0 * r_energy
    total_rewards = total_rewards * r_composition_matching.float()
    return total_rewards


def custom_reward(batch_gen: CrystalBatch) -> torch.Tensor:
    """Calculate custom reward for a batch of structures."""
    gen_structures = batch_gen.to_structure()
    # Placeholder implementation - replace with actual custom reward calculation
    return torch.ones(len(gen_structures), dtype=torch.float32)


def reward_bandgap(
    batch_gen: CrystalBatch, predictor, target_value=None
) -> torch.Tensor:
    device = predictor.device
    pred = predictor.predict(batch_gen.to(device))
    pred = pred["dft_band_gap"].clamp(min=0.0)
    if target_value is not None:
        pred = -((pred - target_value) ** 2)
    r_bandgap = normalize(pred)

    # For diversity reward (This part is optional, can be removed if not needed)
    m = Metrics(metrics=["composition_diversity"])
    assert m._reference_composition_features is not None
    ref_composition_features = m._reference_composition_features.to(device)
    gen_structures = batch_gen.to_structure()
    gen_features = featurize(gen_structures)
    gen_composition_features = gen_features["composition_features"].to(device)
    r_composition_diversity = mmd_reward(
        z_gen=gen_composition_features, z_ref=ref_composition_features
    )["r_indiv"]
    r_composition_diversity = normalize(r_composition_diversity)

    # Total rewards
    total_rewards = r_bandgap + 0.5 * r_composition_diversity
    return total_rewards


###############################################################################
#                                  Utils                                      #
###############################################################################


def standardize(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    if x.std() < eps:
        return torch.zeros_like(x)
    return (x - x.mean()) / (x.std() + eps)


def normalize(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    if x.max() - x.min() < eps:
        return torch.zeros_like(x)
    x = (x - x.min()) / (x.max() - x.min() + eps)
    return x.clamp(0.0, 1.0)


def mmd_reward(z_gen, z_ref):
    """Training Diffusion Models Towards Diverse Image Generation with Reinforcement Learning."""

    def poly_k(z, y, deg=3):
        d = z.size(-1)
        return (z @ y.T / d + 1) ** deg

    M, N = len(z_gen), len(z_ref)

    k_gg = poly_k(z_gen, z_gen)
    k_rr = poly_k(z_ref, z_ref)
    k_gr = poly_k(z_gen, z_ref)

    # Compute MMD
    R_term = (k_rr.sum() - k_rr.trace()) / (N * (N - 1))
    G = k_gg.sum() - k_gg.trace()
    C = k_gr.sum()
    mmd_full = G / (M * (M - 1)) + R_term - 2 * C / (M * N)  # Eq. (11)

    # Compute individual MMD
    S = k_gg.sum(dim=1) - k_gg.diagonal()  # S_m
    T = k_gr.sum(dim=1)  # T_m

    Mp = M - 1
    Ap = Mp * (Mp - 1)
    mmd_drop = (G - 2 * S) / Ap + R_term - 2 * (C - T) / (Mp * N)

    r_indiv = mmd_drop - mmd_full
    return {"r": -mmd_full, "r_indiv": r_indiv}
