"""Compute MLED scores for chemeleon2 samples."""

import gzip
import pickle

import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from xtalmet.constants import HF_VERSION

from compute_mled import LocalEnvironmentDiversityCalculator

RESULTS_DIR = "."
model = "chemeleon2"

# Download generated crystals
print(f"Downloading {model} samples...")
path = hf_hub_download(
	repo_id="masahiro-negishi/xtalmet",
	filename=f"mp20/model/{model}.pkl.gz",
	repo_type="dataset",
	revision=HF_VERSION,
)
with gzip.open(path, "rb") as f:
	samples = pickle.load(f)

print(f"Loaded {len(samples)} samples")

# Initialize calculator
calc = LocalEnvironmentDiversityCalculator()

# Compute MLED scores
mled_scores = []
for structure in tqdm(samples, desc="Computing MLED"):
	try:
		features, ent_pos, ent_poly = calc.featurize_structure(structure)
		mled_scores.append(ent_pos + ent_poly)
	except Exception as e:
		print(f"Failed: {e}")
		mled_scores.append(np.nan)

mled_scores = np.array(mled_scores)

# Save
output_path = f"{RESULTS_DIR}/{model}_mled.pkl.gz"
with gzip.open(output_path, "wb") as f:
	pickle.dump(mled_scores, f)

print(f"Saved {len(mled_scores)} MLED scores to {output_path}")
print(f"Mean: {np.nanmean(mled_scores):.4f}, Std: {np.nanstd(mled_scores):.4f}")
