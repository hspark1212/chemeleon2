"""
modified from
https://github.com/facebookresearch/all-atom-diffusion-transformer
"""

import math
from typing import Dict

import torch
from torch import nn
from torch_geometric.utils import to_dense_batch

from src.vae_module.positional_embeddings import (
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)




# def get_index_embedding(indices, emb_dim, max_len=2048):
#     """Creates sine / cosine positional embeddings from a prespecified indices.

#     Args:
#         indices: offsets of size [..., num_tokens] of type integer
#         emb_dim: dimension of the embeddings to create
#         max_len: maximum length

#     Returns:
#         positional embedding of shape [..., num_tokens, emb_dim]
#     """
#     K = torch.arange(emb_dim // 2, device=indices.device)
#     pos_embedding_sin = torch.sin(
#         indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
#     ).to(indices.device)
#     pos_embedding_cos = torch.cos(
#         indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
#     ).to(indices.device)
#     pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
#     return pos_embedding


class TransformerEncoder(nn.Module):
    """Transformer encoder as part of standard Transformer-based VAEs.

    Args:
        max_num_elements: Maximum number of elements in the dataset
        d_model: Dimension of the model
        nhead: Number of attention heads
        dim_feedforward: Dimension of the feedforward network
        activation: Activation function to use
        dropout: Dropout rate
        norm_first: Whether to use pre-normalization in Transformer blocks
        bias: Whether to use bias
        num_layers: Number of layers
    """

    def __init__(
        self,
        max_num_elements=100,
        d_model: int = 1024,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_first: bool = True,
        bias: bool = True,
        num_layers: int = 6,
        max_num_atoms: int = 512,
        index_embedding: str = 'sinusoidal',
    ):
        super().__init__()

        self.max_num_elements = max_num_elements
        self.d_model = d_model
        self.num_layers = num_layers
        self.atom_type_embedder = nn.Embedding(max_num_elements, d_model)
        self.lattices_embedder = nn.Sequential(
            nn.Linear(9, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.frac_coords_embedder = nn.Sequential(
            nn.Linear(3, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        activation = {
            "gelu": nn.GELU(approximate="tanh"),
            "relu": nn.ReLU(),
        }[activation]
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=activation,
                dropout=dropout,
                batch_first=True,
                norm_first=norm_first,
                bias=bias,
            ),
            norm=nn.LayerNorm(d_model),
            num_layers=num_layers,
        )

        self.index_embedding_type = index_embedding
        if index_embedding == 'sinusoidal':
            self.index_embedding = SinusoidalPositionalEmbedding()
        elif index_embedding == 'none' or index_embedding is None:
            self.index_embedding = NoPositionalEmbedding()
        elif index_embedding == 'global_num_atoms':
            self.index_embedding = GlobalNumAtomsEmbedding(
                embedding_dim=d_model,
                mode='sinusoidal',
            )
        else:
            raise ValueError(f"Unknown index_embedding: {index_embedding}")

    @property
    def hidden_dim(self):
        return self.d_model

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: Data object with the following attributes:
                atom_types (torch.Tensor): Atomic numbers of atoms in the batch
                pos (torch.Tensor): Cartesian coordinates of atoms in the batch
                frac_coords (torch.Tensor): Fractional coordinates of atoms in the batch
                cell (torch.Tensor): Lattice vectors of the unit cell
                lattices (torch.Tensor): Lattice parameters of the unit cell (lengths and angles)
                lengths (torch.Tensor): Lengths of the lattice vectors
                angles (torch.Tensor): Angles between the lattice vectors
                num_atoms (torch.Tensor): Number of atoms in the batch
                batch (torch.Tensor): Batch index for each atom
        """
        x = self.atom_type_embedder(batch.atom_types)
        x += self.lattices_embedder(batch.lattices.view(-1, 9))[batch.batch]
        x += self.frac_coords_embedder(batch.frac_coords)

        # Positional embedding
        if self.index_embedding_type == 'none' or self.index_embedding_type is None:
            pass
        elif self.index_embedding_type == 'global_num_atoms':
            x += self.index_embedding(
                batch["num_atoms"], batch["num_atoms"],
                #scatter_mean(encoded_batch["batch"], encoded_batch["batch"], dim=0).to(torch.long) + 1,
            )
        else:
            x += self.index_embedding(batch["token_idx"], self.d_model)

        # Convert from PyG batch to dense batch with padding
        x, token_mask = to_dense_batch(x, batch.batch)

        # Transformer forward pass
        x = self.transformer.forward(x, src_key_padding_mask=(~token_mask))
        x = x[token_mask]

        return {
            "x": x,
            "num_atoms": batch.num_atoms,
            "batch": batch.batch,
            "token_idx": batch.token_idx,
        }
