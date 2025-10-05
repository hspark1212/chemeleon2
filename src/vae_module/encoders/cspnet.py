"""CSPNet encoder architecture for crystal structure encoding.

Adapted from https://github.com/jiaor17/DiffCSP
"""

import math

import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse

from src.utils.scatter import scatter_mean


class SinusoidalTimeEmbeddings(nn.Module):
    """Attention is all you need."""

    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SinusoidsEmbedding(nn.Module):
    """Embedding for periodic distance features."""

    def __init__(self, n_frequencies=10, n_space=3) -> None:
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


class CSPLayer(nn.Module):
    """Message passing layer for cspnet."""

    def __init__(
        self, hidden_dim=128, act_fn=nn.SiLU(), dis_emb=None, ln=False, ip=True
    ) -> None:
        super().__init__()

        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.ip = ip
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 9 + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def edge_model(
        self,
        node_features,
        frac_coords,
        lattices,
        edge_index,
        edge2graph,
        frac_diff=None,
    ):
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        if frac_diff is None:
            xi, xj = frac_coords[edge_index[0]], frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.0
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)  # fourier transform
        if self.ip:
            lattice_ips = lattices @ lattices.transpose(-1, -2)
        else:
            lattice_ips = lattices
        lattice_ips_flatten = lattice_ips.view(-1, 9)
        lattice_ips_flatten_edges = lattice_ips_flatten[edge2graph]
        edges_input = torch.cat([hi, hj, lattice_ips_flatten_edges, frac_diff], dim=1)
        edge_features = self.edge_mlp(edges_input)
        return edge_features

    def node_model(self, node_features, edge_features, edge_index):
        agg = scatter_mean(
            edge_features,
            edge_index[0],
            dim=0,
            dim_size=node_features.shape[0],
        )
        agg = torch.cat([node_features, agg], dim=1)
        out = self.node_mlp(agg)
        return out

    def forward(
        self,
        *,
        node_features,
        lattices,
        frac_coords,
        edge_index,
        edge2graph,
        frac_diff=None,
    ):
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(
            node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff
        )
        node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output


class CSPNet(nn.Module):
    """CSPNet model, adopted from DiffCSP.

    - edge_style = fc
    """

    def __init__(
        self,
        max_num_elements=100,
        hidden_dim=512,
        num_layers=6,
        num_freqs=128,
        ln=True,
        ip=True,
        smooth=False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Node embedding
        if smooth:
            self.node_embedding = nn.Linear(max_num_elements, hidden_dim)
        else:
            self.node_embedding = nn.Embedding(max_num_elements, hidden_dim)

        # Set up activation function
        self.act_fn = nn.SiLU()

        # Set up distance embedding
        self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs)

        # Set up layers
        self.num_layers = num_layers
        self.ln = ln
        self.ip = ip
        for i in range(0, num_layers):
            self.add_module(
                f"csp_layer_{i}",
                CSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln=ln, ip=ip),
            )
        if ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def gen_edges(self, num_atoms, frac_coords):
        # edge_style = fc
        lis = [torch.ones(n, n, device=num_atoms.device) for n in num_atoms]
        fc_graph = torch.block_diag(*lis)
        fc_edges, _ = dense_to_sparse(fc_graph)
        return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.0

    def forward(self, batch):
        # Get attributes
        atom_types = batch.atom_types
        lattices = batch.lattices
        frac_coords = batch.frac_coords
        num_atoms = batch.num_atoms
        batch_idx = batch.batch

        edges, frac_diff = self.gen_edges(num_atoms, frac_coords)
        edge2graph = batch_idx[edges[0]]
        node_features = self.node_embedding(atom_types)  # [B_n, hidden_dim]

        for i in range(0, self.num_layers):
            csp_layer = self._modules[f"csp_layer_{i}"]
            assert csp_layer is not None
            node_features = csp_layer(
                node_features=node_features,
                lattices=lattices,
                frac_coords=frac_coords,
                edge_index=edges,
                edge2graph=edge2graph,
                frac_diff=frac_diff,
            )

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        node_features = self.fc_out(node_features)

        return {
            "x": node_features,
            "num_atoms": batch.num_atoms,
            "batch": batch.batch,
            "token_idx": batch.token_idx,
        }
