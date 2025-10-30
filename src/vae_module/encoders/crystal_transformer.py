"""Periodic O(3) invariant R-Transformer for crystal systems."""

import math

import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from torch_geometric.utils import softmax as scatter_softmax

from src.data.schema import CrystalBatch


class SinusoidsEmbedding(nn.Module):
    """Sinusoidal positional embedding for Fourier features."""

    def __init__(self, n_frequencies=100, n_space=3) -> None:
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SparseTransformerLayer(nn.Module):
    """Sparse Periodic R-Trans with edge-based pair-specific attention."""

    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Projections for Q, K, V (input is [h_n; h_e] with dim 2*hidden_dim)
        self.W_q = nn.Linear(2 * hidden_dim, hidden_dim)  # Eq. 5
        self.W_kv = nn.Linear(2 * hidden_dim, 2 * hidden_dim)  # Eq. 6

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        # Residual + LayerNorm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Feed-forward network (2-layer MLP with GELU)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self, h_n: torch.Tensor, h_e: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with sparse pair-specific attention.

        Args:
            h_n: Node embeddings (B_n, d) - batched atoms
            h_e: Edge embeddings (B_e, d) - sparse edges only
            edge_index: Edge connectivity (2, B_e) - [sources, targets]

        Returns:
            Updated node embeddings (B_e, d)
        """
        num_nodes = h_n.shape[0]
        src, dst = edge_index[0], edge_index[1]  # (B_e,) (B_e,)

        # Gather node features for each edge
        h_n_src = h_n[src]  # (B_e, d)
        h_n_dst = h_n[dst]  # (B_e, d)

        # Concatenate node and edge features - Eq. 5, 6
        q_input = torch.cat([h_n_src, h_e], dim=-1)  # (B_e, 2d)
        kv_input = torch.cat([h_n_dst, h_e], dim=-1)  # (B_e, 2d)

        # Project to Q, K, V (edge-wise)
        Q = self.W_q(q_input)  # (B_e, d)
        KV = self.W_kv(kv_input)  # (B_e, 2d)
        K, V = KV.chunk(2, dim=-1)  # (B_e, d), (B_e, d)

        # Reshape for multi-head attention: (B_e, d) -> (B_e, H, d_h)
        Q = Q.view(-1, self.num_heads, self.head_dim)
        K = K.view(-1, self.num_heads, self.head_dim)
        V = V.view(-1, self.num_heads, self.head_dim)

        # Compute attention scores - Eq. 7 (edge-wise dot product)
        attn_scores = (Q * K).sum(dim=-1) / math.sqrt(self.head_dim)  # (B_e, H)

        # Apply softmax per source node - normalized over all targets j for each source i
        attn_weights = scatter_softmax(attn_scores, src, dim=0)  # (B_e, H)

        # Weighted aggregation - Eq. 8
        attn_weights = attn_weights.unsqueeze(-1)  # (B_e, H, 1)
        weighted_values = attn_weights * V  # (B_e, H, d_h)

        # Aggregate messages for each node (sum over incoming edges)
        out = scatter(
            weighted_values, src, dim=0, dim_size=num_nodes, reduce="sum"
        )  # (B_n, H, d_h)

        # Concatenate heads and project
        out = out.view(num_nodes, self.num_heads * self.head_dim)  # (B_n, d)
        out = self.output_proj(out)

        # Residual connection + LayerNorm - Eq. 9
        h_n = self.norm1(h_n + self.dropout1(out))

        # Feed-forward network with residual
        h_n = self.norm2(h_n + self.dropout2(self.ffn(h_n)))

        return h_n


class CrystalTransformerEncoder(nn.Module):
    """Periodic R-Transformer encoder for crystal systems."""

    def __init__(
        self,
        max_num_elements=100,
        d_model: int = 256,
        n_frequencies: int = 100,
        nhead: int = 4,
        num_layers: int = 6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.max_num_elements = max_num_elements
        self.d_model = d_model

        # Embeddings
        self.atom_embedding = nn.Embedding(max_num_elements, d_model)
        self.fourier_embedding = SinusoidsEmbedding(n_frequencies=n_frequencies)

        # Input projection
        self.node_proj = nn.Linear(d_model + 6, d_model)
        self.edge_proj = nn.Linear(self.fourier_embedding.dim, d_model)

        # Sparse Transformer layers
        self.layers = nn.ModuleList(
            [
                SparseTransformerLayer(
                    hidden_dim=d_model, num_heads=nhead, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

        # Final normalization
        self.final_norm = nn.LayerNorm(d_model)

    @property
    def hidden_dim(self) -> int:
        return self.d_model

    def get_lattice_invariant_features(self, l: torch.Tensor) -> torch.Tensor:
        LTL = torch.bmm(l.transpose(1, 2), l)  # Gram matrix
        i, j = torch.triu_indices(3, 3)
        return LTL[:, i, j]

    def get_relative_edge_features(
        self,
        batch: CrystalBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Get pre-computed edges from batch
        edge_index = batch.edge_index  # (2, E)
        edge_unit_shifts = batch.edge_unit_shifts  # (B_e, 3)

        # Get source and destination node indices
        src, dst = edge_index[0], edge_index[1]  # (B_e,)

        # Compute fractional displacements with PBC
        frac_src = batch.frac_coords[src]  # (B_e, 3)
        frac_dst = batch.frac_coords[dst]  # (B_e, 3)
        edge_frac_disp = (frac_dst - frac_src) + edge_unit_shifts  # (B_e, 3)

        # Encode edge features using Fourier embedding
        edge_frac_disp = edge_frac_disp % 1.0  # Wrap into [0, 1)
        edge_feat = self.fourier_embedding(edge_frac_disp)  # (B_e, F)

        return edge_feat

    def forward(self, batch: CrystalBatch):
        # Node embeddings
        e_a = self.atom_embedding(batch.atom_types)  # (B_n, d)
        l_feat = self.get_lattice_invariant_features(batch.lattices)  # (B, 6)
        node_feat = torch.cat([e_a, l_feat[batch.batch]], dim=-1)  # (B_n, d+6)
        h_n = self.node_proj(node_feat)  # (B_n, d)

        # Edge embeddings
        edge_feat = self.get_relative_edge_features(batch)  # (2, E), (B_e, F)
        h_e = self.edge_proj(edge_feat)  # (B_e, d)

        # Crystal Transformer layers
        for layer in self.layers:
            h_n = layer(h_n, h_e, batch.edge_index)  # (B_n, d)

        # Final normalization
        h_n = self.final_norm(h_n)

        return {
            "x": h_n,
            "num_atoms": batch.num_atoms,
            "batch": batch.batch,
            "token_idx": batch.token_idx,
        }
