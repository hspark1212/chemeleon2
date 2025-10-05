"""Conditioning modules for the latent diffusion model.

This module provides encoders for various conditioning types including compositions,
chemical systems, numerical values, categorical labels, and text prompts.
"""

from enum import Enum

import torch
import torch.nn as nn
from pymatgen.core import Composition, Element


class ConditionType(Enum):
    """Enum for different types of conditions."""

    COMPOSITION = "composition"  # chemical formula (e.g., "LiMnO2")
    CHEMICAL_SYSTEM = "chemical_system"  # chemical system (e.g., "Li-Mn-O")
    VALUE = "float"  # numerical value (e.g., 0.5, 1.3)
    CATEGORICAL = "categorical"  # categorical class (e.g., 1, 2, 3)
    TEXT = "text"  # text prompt (e.g., "crystal structure of LiMnO2")


class ConditionModule(nn.Module):
    """Conditional encoding module for latent diffusion."""

    def __init__(
        self,
        condition_type: dict,
        hidden_dim: int,
        drop_prob: float,
        stats: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.condition_type = condition_type
        self.target_condition = list(condition_type.keys())
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.stats = stats if stats is not None else {}

        self.encoders = nn.ModuleDict({})
        for cond_name, cond_type in condition_type.items():
            if cond_type == ConditionType.COMPOSITION.value:
                self.encoders[cond_name] = CompositionEncoder(
                    in_dim=100,  # max number of elements
                    hidden_dim=hidden_dim,
                )
            elif cond_type == ConditionType.CHEMICAL_SYSTEM.value:
                self.encoders[cond_name] = ChemicalSystemEncoder(
                    in_dim=100,  # max number of elements
                    hidden_dim=hidden_dim,
                )
            elif cond_type == ConditionType.VALUE.value:
                _stats = self.stats.get(cond_name, {})
                self.encoders[cond_name] = ValueEncoder(
                    hidden_dim=hidden_dim,
                    mean=_stats.get("mean", None),
                    std=_stats.get("std", None),
                )
            elif cond_type == ConditionType.CATEGORICAL.value:
                assert "num_classes" in kwargs, (
                    "num_classes must be provided when using CLASS condition type"
                )
                self.encoders[cond_name] = CategoricalEncoder(
                    in_dim=kwargs["num_classes"],
                    hidden_dim=hidden_dim,
                )
            elif cond_type == ConditionType.TEXT.value:
                self.encoders[cond_name] = TextEncoder(
                    hidden_dim=hidden_dim,
                )
            else:
                raise ValueError(f"Unsupported condition type: {cond_type}")

        self.proj = nn.Sequential(
            nn.Linear(len(self.encoders) * hidden_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )

    def forward(self, batch_y, training=True):
        """Forward pass for the condition module.

        - In training mode, randomly drops conditions for classifier-free guidance.
        - In inference mode, duplicates inputs to create conditioned and unconditioned pairs.

        :returns: Embeddings of shape (B, L) during training, or (2*B, L) during inference.
        """
        target_conditions = list(batch_y.keys())
        assert set(target_conditions) == set(self.target_condition), (
            f"Expected conditions {self.target_condition}, but got {target_conditions}"
        )

        batch_size = len(list(batch_y.values())[0])
        if training:
            assert self.drop_prob >= 0
            drop_mask = torch.rand(batch_size, device=self.device) < self.drop_prob
        else:
            drop_mask = torch.cat([torch.zeros(batch_size), torch.ones(batch_size)])
            drop_mask = drop_mask.bool().to(self.device)
            batch_y = {k: _duplicate(v) for k, v in batch_y.items()}

        cond_embeds = []
        for cond_name, encoder in self.encoders.items():
            y = batch_y[cond_name]
            embed = encoder(y, drop_mask)
            cond_embeds.append(embed)

        cond_embeds = torch.cat(cond_embeds, dim=-1)
        cond_embeds = self.proj(cond_embeds)
        return cond_embeds

    @property
    def device(self):
        return next(self.parameters()).device


def _duplicate(v):
    if isinstance(v, torch.Tensor):
        return torch.cat([v, v])
    elif isinstance(v, list):
        return v * 2
    else:
        raise ValueError(f"Unsupported type: {type(v)}")


#########################################################################
#                               Encoders                                #
#########################################################################
class BaseEncoder(nn.Module):
    """Base encoder class for conditioning modalities."""

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        preprocess,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.preprocess = preprocess
        self.embedding = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.null_embed = nn.Parameter(torch.randn(1, in_dim))

    def forward(self, y, drop_mask=None):
        y = self.preprocess(y).to(self.device)
        if drop_mask is not None:
            y[drop_mask] = self.null_embed
        return self.embedding(y)

    @property
    def device(self):
        """Get device from module parameters."""
        return next(self.parameters()).device


class ValueEncoder(BaseEncoder):
    """Encoder for continuous scalar values."""

    def __init__(self, hidden_dim, mean=None, std=None) -> None:
        def preprocess(batch):
            batch = torch.as_tensor(batch).float().unsqueeze(-1)
            if mean is not None and std is not None:
                batch = (batch - mean) / std
            return batch

        super().__init__(
            in_dim=1,
            hidden_dim=hidden_dim,
            preprocess=preprocess,
        )


class CategoricalEncoder(BaseEncoder):
    """Encoder for categorical variables."""

    def __init__(self, in_dim, hidden_dim) -> None:
        def preprocess(batch):
            idx = torch.as_tensor(batch).long()
            return torch.nn.functional.one_hot(idx, num_classes=in_dim).float()

        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            preprocess=preprocess,
        )


class CompositionEncoder(BaseEncoder):
    """Encoder for chemical composition."""

    def __init__(self, in_dim, hidden_dim) -> None:
        def preprocess(batch):
            vals = [self._composition_to_embeds(comp_str) for comp_str in batch]
            return torch.cat(vals, dim=0)

        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            preprocess=preprocess,
        )

    def _composition_to_embeds(self, comp_str: str):
        v = torch.zeros(self.in_dim)
        comp = Composition(comp_str).reduced_composition  # Use reduced composition
        for el, amt in comp.get_el_amt_dict().items():
            v[Element(el).Z] = float(amt)
        v = v / v.sum() if v.sum() > 0 else v  # Normalize
        return v.unsqueeze(0)


class ChemicalSystemEncoder(BaseEncoder):
    """Encoder for chemical system representation."""

    def __init__(self, in_dim, hidden_dim) -> None:
        def preprocess(batch):
            vals = [self._chemical_system_to_embeds(cs) for cs in batch]
            return torch.cat(vals, dim=0)

        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            preprocess=preprocess,
        )

    def _chemical_system_to_embeds(self, cs: str):
        v = torch.zeros(self.in_dim)
        elements = cs.split("-")
        for el in elements:
            v[Element(el).Z] = 1.0
        return v.unsqueeze(0)


class TextEncoder(BaseEncoder):
    """Placeholder text encoder for future text conditioning."""

    def __init__(self, hidden_dim) -> None:
        def preprocess(batch):
            return torch.zeros(len(batch), 100)  # Dummy tensor

        super().__init__(
            in_dim=100,  # Assuming a fixed input dimension for text
            hidden_dim=hidden_dim,
            preprocess=preprocess,
        )
