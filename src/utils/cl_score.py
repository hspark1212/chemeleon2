"""Crystal Lattice score calculation utilities.

Adapted from https://github.com/snu-micc/Synthesizability-PU-CGCNN
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
from pymatgen.core import Structure
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

BENCHMARK_DIR = Path(__file__).resolve().parent.parent.parent / "benchmarks"
CLSCORE_MODEL_DIR = BENCHMARK_DIR / "cl_score_trained_models"
ATOM_INIT_FILE = CLSCORE_MODEL_DIR / "atom_init.json"
max_num_nbr = 12
radius = 8
num_models = 100


def download_trained_models():
    """Download trained models for CLScore from the GitHub repository."""
    if not CLSCORE_MODEL_DIR.exists():
        CLSCORE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {CLSCORE_MODEL_DIR}")

    # Skip download if the models already exist
    if len(list(CLSCORE_MODEL_DIR.glob("checkpoint_bag_*.pth.tar"))) == num_models:
        return

    print(
        "Downloading trained models for CLScore from snu-micc/Synthesizability-PU-CGCNN..."
    )
    api = (
        "https://api.github.com/repos/"
        "snu-micc/Synthesizability-PU-CGCNN/contents/trained_models"
    )

    for item in requests.get(api, timeout=60).json():
        if item["type"] == "file":
            fname = os.path.join(CLSCORE_MODEL_DIR, item["name"])
            with requests.get(item["download_url"], stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(fname, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
    print(f"Successfully downloaded trained models to {CLSCORE_MODEL_DIR}")


def download_atom_init_file():
    """Download the atom initialization file for CLScore."""
    if not os.path.exists(ATOM_INIT_FILE):
        print(f"Downloading atom initialization file to {ATOM_INIT_FILE}...")
        url = "https://raw.githubusercontent.com/txie-93/cgcnn/refs/heads/master/data/sample-regression/atom_init.json"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(ATOM_INIT_FILE, "w") as f:
            f.write(response.text)
        print("Atom initialization file downloaded successfully.")
    else:
        pass


###############################################################################
#                             Utility Functions                              #
###############################################################################


class GaussianDistance:
    """Gaussian distance expansion for structure features."""

    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(-((distances[..., np.newaxis] - self.filter) ** 2) / self.var**2)


class AtomInitializer:
    """Base class for atomic feature initialization."""

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}
        self._decodedict = None

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {
            idx: atom_type for atom_type, idx in self._embedding.items()
        }

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if self._decodedict is None:
            self._decodedict = {
                idx: atom_type for atom_type, idx in self._embedding.items()
            }
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """Atomic feature initializer from JSON file."""

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


def create_crystal_graph(
    structure: Structure,
    radius,
    max_num_nbr,
    ari: AtomInitializer,
    gdf: GaussianDistance,
):
    atom_fea = np.vstack(
        [ari.get_atom_fea(structure[j].specie.number) for j in range(len(structure))]  # type: ignore[union-attr]
    )
    atom_fea = torch.Tensor(atom_fea)
    all_nbrs = structure.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr_fea_idx, nbr_fea = [], []
    for nbr in all_nbrs:
        if len(nbr) < max_num_nbr:
            nbr_fea_idx.append(
                list(map(lambda x: x[2], nbr)) + [0] * (max_num_nbr - len(nbr))
            )
            nbr_fea.append(
                list(map(lambda x: x[1], nbr))
                + [radius + 1.0] * (max_num_nbr - len(nbr))
            )
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:max_num_nbr])))
            nbr_fea.append(list(map(lambda x: x[1], nbr[:max_num_nbr])))
    nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
    nbr_fea = gdf.expand(nbr_fea)
    atom_fea = torch.Tensor(atom_fea)
    nbr_fea = torch.Tensor(nbr_fea)
    nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
    preload_data = (atom_fea, nbr_fea, nbr_fea_idx)
    return preload_data


def collate_pool(dataset_list):
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx = []
    base_idx = 0
    for i, (atom_fea, nbr_fea, nbr_fea_idx) in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
        crystal_atom_idx.append(new_idx)
        base_idx += n_i
    return (
        torch.cat(batch_atom_fea, dim=0),
        torch.cat(batch_nbr_fea, dim=0),
        torch.cat(batch_nbr_fea_idx, dim=0),
        crystal_atom_idx,
    )


class ConvLayer(nn.Module):
    """Convolutional layer for crystal graph neural network."""

    def __init__(self, atom_fea_len, nbr_fea_len):
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(
            2 * self.atom_fea_len + self.nbr_fea_len, 2 * self.atom_fea_len
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        # Convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [
                atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
                atom_nbr_fea,
                nbr_fea,
            ],
            dim=2,
        )
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(
            total_gated_fea.view(-1, self.atom_fea_len * 2)
        ).view(N, M, self.atom_fea_len * 2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGraphConvNet(nn.Module):
    """Crystal Graph Convolutional Neural Network for property prediction."""

    def __init__(
        self,
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
        n_h=1,
        classification=False,
    ):
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList(
            [
                ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
                for _ in range(n_conv)
            ]
        )
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        self.final_fea = 0

        if n_h > 1:
            self.fcs = nn.ModuleList(
                [nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)]
            )
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, "fcs") and hasattr(self, "softpluses"):
            for fc, softplus in zip(self.fcs, self.softpluses, strict=False):
                crys_fea = softplus(fc(crys_fea))

        self.final_fea = crys_fea

        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        assert (
            sum([len(idx_map) for idx_map in crystal_atom_idx])
            == atom_fea.data.shape[0]
        )
        summed_fea = [
            torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
            for idx_map in crystal_atom_idx
        ]
        return torch.cat(summed_fea, dim=0)


def predict_PU_learning(model, test_loader, use_cuda):
    model.eval()
    if use_cuda:
        model.cuda()

    test_preds = []
    for i, input in enumerate(test_loader):
        with torch.no_grad():
            if use_cuda:
                input_var = (
                    Variable(input[0].cuda(non_blocking=True)),
                    Variable(input[1].cuda(non_blocking=True)),
                    input[2].cuda(non_blocking=True),
                    [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]],
                )
            else:
                input_var = (Variable(input[0]), Variable(input[1]), input[2], input[3])

        output = model(*input_var)
        test_pred = torch.exp(output.data.cpu())
        assert test_pred.shape[1] == 2
        test_preds += test_pred[:, 1].tolist()
    return test_preds


###############################################################################
#                             Main Function to Compute CLscore                #
###############################################################################


def compute_clscore(structures: list[Structure], batch_size=100, use_cuda=True):
    download_trained_models()
    download_atom_init_file()
    atom_init_file = Path(ATOM_INIT_FILE)
    if not atom_init_file.exists():
        raise FileNotFoundError(f"Atom initialization file not found: {atom_init_file}")
    ari = AtomCustomJSONInitializer(atom_init_file)
    gdf = GaussianDistance(dmin=0.0, dmax=radius, step=0.2)
    data_list = [
        create_crystal_graph(
            st,
            radius=radius,
            max_num_nbr=max_num_nbr,
            ari=ari,
            gdf=gdf,
        )
        for i, st in tqdm(
            enumerate(structures), total=len(structures), desc="Creating crystal graphs"
        )
    ]
    test_loader = DataLoader(
        data_list,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_pool,  # type: ignore[arg-type]
    )
    total_results = []

    # Check if the model directory exists
    model_dir = Path(CLSCORE_MODEL_DIR)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    # Check the number of model files
    model_files = list(model_dir.glob("checkpoint_bag_*.pth.tar"))
    assert len(model_files) == num_models, (
        f"Expected {num_models} model files, found {len(model_files)}."
    )

    for i in tqdm(range(1, num_models + 1)):
        model_path = Path(model_dir) / f"checkpoint_bag_{i}.pth.tar"
        assert model_path.exists(), f"Model file {model_path} does not exist."
        model_checkpoint = torch.load(
            model_path, map_location=lambda storage, loc: storage, weights_only=False
        )
        model_args = argparse.Namespace(**model_checkpoint["args"])
        # build model
        input_ = data_list[0]
        orig_atom_fea_len = input_[0].shape[-1]
        nbr_fea_len = input_[1].shape[-1]
        model = CrystalGraphConvNet(
            orig_atom_fea_len,
            nbr_fea_len,
            atom_fea_len=model_args.atom_fea_len,
            n_conv=model_args.n_conv,
            h_fea_len=model_args.h_fea_len,
            n_h=model_args.n_h,
            classification=True,
        )
        if os.path.isfile(model_path):
            checkpoint = torch.load(
                model_path,
                map_location=lambda storage, loc: storage,
                weights_only=False,
            )
            model.load_state_dict(checkpoint["state_dict"])
        preds = predict_PU_learning(model, test_loader, use_cuda)
        total_results.append(preds)
    total_results = np.array(total_results)
    clscore = np.mean(total_results, axis=0)
    return clscore
