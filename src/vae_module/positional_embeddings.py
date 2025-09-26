"""Reusable positional embedding components for VAEs."""

from __future__ import annotations
import math
import torch
from torch import Tensor, nn


class PositionalEmbedding(nn.Module):
    """Base interface for positional embeddings."""

    def forward(self, indices: Tensor, emb_dim: int) -> Tensor:  # pragma: no cover
        raise NotImplementedError


class SinusoidalPositionalEmbedding(PositionalEmbedding):
    """Deterministic sine/cosine encoding (default)."""

    def __init__(self, max_len: int = 2048) -> None:
        super().__init__()
        self.max_len = max_len

    def forward(self, indices: Tensor, emb_dim: int) -> Tensor:
        if emb_dim % 2:
            raise ValueError("Sinusoidal positional embeddings require even emb_dim.")
        device = indices.device
        indices = indices.to(torch.float32)
        k = torch.arange(emb_dim // 2, device=device, dtype=torch.float32)
        div_term = torch.pow(self.max_len, 2 * k / emb_dim)
        angles = indices[..., None] * math.pi / div_term
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class LearnedPositionalEmbedding(PositionalEmbedding):
    """Simple learnable lookup table."""

    def __init__(self, embedding_dim: int, max_len: int = 512) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_len, embedding_dim)

    def forward(self, indices: Tensor, emb_dim: int) -> Tensor:
        if emb_dim != self.embedding.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding.embedding_dim}, got {emb_dim}"
            )
        return self.embedding(indices)


class NoPositionalEmbedding(PositionalEmbedding):
    """Returns zeros to disable positional information."""

    def forward(self, indices: Tensor, emb_dim: int) -> Tensor:
        return torch.zeros((*indices.shape, emb_dim), device=indices.device)


class GlobalNumAtomsEmbedding(nn.Module):
    """
    Embed a per-graph scalar (number of atoms) and broadcast it to all nodes.

    Args:
        embedding_dim: Output feature dimension.
        mode: "learned" (nn.Embedding) or "sinusoidal".
        max_value: Max integer value expected for num_atoms (only for "learned").
        sinusoidal_max_len: Base used by the sinusoidal encoder (only for "sinusoidal").

    Shapes:
        num_atoms: Long tensor, shape (B,) with number of atoms per graph.
        num_nodes_per_graph: Long tensor, shape (B,) with node counts per graph.

    Returns:
        Tensor of shape (sum(num_nodes_per_graph), embedding_dim), containing the
        same embedding repeated for all nodes belonging to the same graph.
    """

    def __init__(
        self,
        embedding_dim: int,
        mode: str = "sinusoidal",
        max_value: int = 8192,
        sinusoidal_max_len: int = 2048,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mode = mode.lower()
        if self.mode == "learned":
            # +1 so value==max_value is in-bounds
            self.table = nn.Embedding(max_value + 1, embedding_dim)
        elif self.mode == "sinusoidal":
            # Reuse the sinusoidal positional embedding defined above
            self.sine = SinusoidalPositionalEmbedding(max_len=sinusoidal_max_len)
            if embedding_dim % 2 != 0:
                raise ValueError("Sinusoidal mode requires an even embedding_dim.")
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'learned' or 'sinusoidal'.")

    def forward(self, num_atoms: Tensor, num_nodes_per_graph: Tensor) -> Tensor:
        if num_atoms.dim() != 1 or num_nodes_per_graph.dim() != 1:
            raise ValueError("num_atoms and num_nodes_per_graph must be 1D tensors.")
        if num_atoms.shape[0] != num_nodes_per_graph.shape[0]:
            raise ValueError("num_atoms and num_nodes_per_graph must have same length (B).")

        num_atoms = num_atoms.to(dtype=torch.long, device=num_nodes_per_graph.device)
        num_nodes_per_graph = num_nodes_per_graph.to(dtype=torch.long)

        # Broadcast per-graph scalar to per-node indices: [10, 4, ...] -> [10,10,...,4,4,...]
        per_node_indices = torch.repeat_interleave(num_atoms, num_nodes_per_graph)

        if self.mode == "learned":
            max_idx = int(per_node_indices.max().item()) if per_node_indices.numel() else -1
            if max_idx >= self.table.num_embeddings:
                raise ValueError(
                    f"num_atoms contains value {max_idx}, but embedding table "
                    f"has only {self.table.num_embeddings} entries. "
                    "Increase max_value when constructing GlobalNumAtomsEmbedding."
                )
            return self.table(per_node_indices)  # (N_total, embedding_dim)

        # Sinusoidal path
        return self.sine(per_node_indices, self.embedding_dim)  # (N_total, embedding_dim)

__all__ = [
    "PositionalEmbedding",
    "SinusoidalPositionalEmbedding",
    "LearnedPositionalEmbedding",
    "NoPositionalEmbedding",
    "GlobalNumAtomsEmbedding",
]
