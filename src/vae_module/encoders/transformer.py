"""modified from
https://github.com/facebookresearch/all-atom-diffusion-transformer.
"""

import math

import torch
from torch import nn
from torch_geometric.utils import to_dense_batch

from src.data.schema import CrystalBatch


def get_index_embedding(indices, emb_dim, max_len=2048):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., num_tokens] of type integer
        emb_dim: dimension of the embeddings to create
        max_len: maximum length

    Returns:
        positional embedding of shape [..., num_tokens, emb_dim]
    """
    K = torch.arange(emb_dim // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], dim=-1)
    return pos_embedding


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
    ) -> None:
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

    @property
    def hidden_dim(self) -> int:
        return self.d_model

    def forward(self, batch: CrystalBatch) -> dict[str, torch.Tensor]:
        """Args:
        batch: Data object with the following attributes:
            atom_types (torch.Tensor): Atomic numbers of atoms in the batch
            pos (torch.Tensor): Cartesian coordinates of atoms in the batch
            frac_coords (torch.Tensor): Fractional coordinates of atoms in the batch
            cell (torch.Tensor): Lattice vectors of the unit cell
            lattices (torch.Tensor): Lattice parameters of the unit cell (lengths and angles)
            lengths (torch.Tensor): Lengths of the lattice vectors
            angles (torch.Tensor): Angles between the lattice vectors
            num_atoms (torch.Tensor): Number of atoms in the batch
            batch (torch.Tensor): Batch index for each atom.
        """
        atom_types = batch.atom_types
        lattices = batch.lattices
        frac_coords = batch.frac_coords
        token_idx = batch.token_idx
        batch_idx = batch.batch
        num_atoms = batch.num_atoms

        x = self.atom_type_embedder(atom_types)
        x += self.lattices_embedder(lattices.view(-1, 9))[batch_idx]
        x += self.frac_coords_embedder(frac_coords)

        # Positional embedding
        x += get_index_embedding(token_idx, self.d_model)

        # Convert from PyG batch to dense batch with padding
        x, token_mask = to_dense_batch(x, batch_idx)

        # Transformer forward pass
        x = self.transformer.forward(x, src_key_padding_mask=(~token_mask))
        x = x[token_mask]

        return {
            "x": x,
            "num_atoms": num_atoms,
            "batch": batch_idx,
            "token_idx": token_idx,
        }
