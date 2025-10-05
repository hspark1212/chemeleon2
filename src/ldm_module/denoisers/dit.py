"""Diffusion Transformer (DiT) denoiser architecture.

This module implements the DiT architecture for denoising in the latent diffusion model,
adapted from Meta's DiT paper with support for variable-length sequences and masking.
"""

# Adapted from:
# https://github.com/facebookresearch/DiT/models.py
# https://github.com/facebookresearch/all-atom-diffusion-transformer/src/models/denoisers/dit.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math
from collections.abc import Callable

import torch
import torch.nn as nn


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_dim, frequency_embedding_dim=256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.frequency_embedding_dim = frequency_embedding_dim

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_dim)
        t_emb = self.mlp(t_freq)
        return t_emb


def get_pos_embedding(indices, emb_dim, max_len=2048):
    """Creates sine / cosine positional embeddings from a prespecified indices."""
    K = torch.arange(emb_dim // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], dim=-1)
    return pos_embedding


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=0, bias=True, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)

        self.mlp = Mlp(
            in_features=hidden_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

    def forward(self, x, c, mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        _x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = (
            x
            + gate_msa.unsqueeze(1)
            * self.attn(_x, _x, _x, key_padding_mask=mask, need_weights=False)[0]
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """The final layer of DiT."""

    def __init__(self, hidden_dim, out_dim) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """Diffusion model with a Transformer backbone."""

    def __init__(
        self,
        *,
        latent_dim=8,  # latent dim
        depth=12,
        hidden_size=384,
        num_heads=6,
        mlp_ratio=4.0,
        learn_sigma=True,
    ) -> None:
        super().__init__()
        self.learn_sigma = learn_sigma
        self.latent_dim = latent_dim
        self.depth = depth
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.x_embedder = nn.Linear(latent_dim, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(
            hidden_size, latent_dim * 2 if learn_sigma else latent_dim
        )
        self.initialize_weights()

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize x_embedder
        nn.init.normal_(self.x_embedder.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)  # type: ignore
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)  # type: ignore

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)  # type: ignore
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)  # type: ignore

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)  # type: ignore
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)  # type: ignore
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, mask, y=None):
        """Forward pass of DiT.

        x: (B, N, L) tensor of latent inputs
        t: (B,) tensor of diffusion timesteps
        mask: (B, N) tensor of masks for latent inputs
        y: (B,) tensor of class labels
        """
        token_indices = torch.cumsum(mask, dim=-1) - 1
        pos_emb = get_pos_embedding(token_indices, self.hidden_size)
        x = self.x_embedder(x) + pos_emb  # (B, N, H)
        t = self.t_embedder(t)  # (B, H)
        y = (
            y
            if y is not None
            else torch.zeros(x.shape[0], self.hidden_size, device=x.device)  # Dummy
        )
        c = t + y
        for block in self.blocks:
            x = block(x, c, ~mask)  # (B, N, H)
        x = self.final_layer(x, c)  # (B, N, L) or (B, N, 2L)
        if self.learn_sigma:
            assert x.shape[2] == 2 * self.latent_dim
            x = x.view(x.shape[0], 2 * x.shape[1], self.latent_dim)  # (B, 2N, L)
            x = x * mask.repeat(1, 2).unsqueeze(-1)  # (B, 2N, L)
        else:
            assert x.shape[2] == self.latent_dim
            x = x * mask.unsqueeze(-1)  # (B, N, L)
        return x

    def forward_with_cfg(self, x, t, mask, y, cfg_scale):
        """Forward pass of DiT, but also batches the unconditional forward pass for classifier-free
        guidance.
        """
        half_x = x[: x.shape[0] // 2]
        combined_x = torch.cat([half_x, half_x], dim=0)
        model_out = self.forward(combined_x, t, mask, y)

        cond_eps, uncond_eps = torch.split(model_out, len(model_out) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return eps


#################################################################################
#                                   DiT Configs                                  #
#################################################################################


def DiT_XL(**kwargs):
    return DiT(depth=28, hidden_size=1152, num_heads=16, **kwargs)


def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)


def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)


def DiT_S(**kwargs):
    return DiT(depth=6, hidden_size=384, num_heads=6, **kwargs)


DiT_models = {
    "DiT-XL": DiT_XL,
    "DiT-L": DiT_L,
    "DiT-B": DiT_B,
    "DiT-S": DiT_S,
}
