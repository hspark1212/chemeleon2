#!/usr/bin/env python3
"""
Compute multi-metric structural novelty for generated crystal structures.

Metrics:
1. Space Group Novelty - How rare is the space group in training set
2. Structure Match Novelty - Does any training structure match closely
3. Coordination Pattern Novelty - How rare is the coordination pattern

Output: chemeleon2_structural_novelty.pkl.gz
"""

import gzip
import io
import pickle
import warnings
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import requests
from huggingface_hub import hf_hub_download
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm
from xtalmet.constants import HF_VERSION


class StructuralNoveltyCalculator:
	"""Multi-metric structural novelty calculator."""

	def __init__(
		self,
		training_structures: list[Structure],
		ltol: float = 0.2,
		stol: float = 0.3,
		angle_tol: float = 5.0,
	):
		"""
		Initialize calculator with training structures.

		Args:
			training_structures: List of training Structure objects
			ltol: Length tolerance for StructureMatcher
			stol: Site tolerance for StructureMatcher
			angle_tol: Angle tolerance for StructureMatcher
		"""
		self.training_structures = training_structures
		self.matcher = StructureMatcher(
			ltol=ltol, stol=stol, angle_tol=angle_tol, primitive_cell=True
		)
		self.crystal_nn = CrystalNN(
			distance_cutoffs=None,
			x_diff_weight=0,
			porous_adjustment=False,
			search_cutoff=12,
		)

		# Pre-compute training set statistics
		print("Pre-computing training set statistics...")
		self._precompute_training_stats()

	def _precompute_training_stats(self):
		"""Precompute space groups and coordination patterns for training set."""
		self.train_space_groups = []
		self.train_cn_patterns = []

		for struct in tqdm(self.training_structures, desc="Analyzing training set"):
			# Space group
			try:
				sg = SpacegroupAnalyzer(struct, symprec=0.1).get_space_group_number()
			except Exception:
				sg = 0
			self.train_space_groups.append(sg)

			# Coordination pattern
			cn_pattern = self._get_cn_pattern(struct)
			self.train_cn_patterns.append(cn_pattern)

		# Count space group frequencies
		self.sg_counter = Counter(self.train_space_groups)
		self.cn_counter = Counter(self.train_cn_patterns)

		print(f"Training set: {len(self.training_structures)} structures")
		print(f"Unique space groups: {len(self.sg_counter)}")
		print(f"Unique CN patterns: {len(self.cn_counter)}")

	def _get_cn_pattern(self, structure: Structure) -> str:
		"""Get coordination number pattern as a sorted string."""
		cns = []
		for i in range(len(structure)):
			try:
				cn = self.crystal_nn.get_cn(structure, i)
				cns.append(int(cn))
			except Exception:
				cns.append(0)
		# Sort and create pattern string
		cns_sorted = tuple(sorted(cns))
		return str(cns_sorted)

	def space_group_novelty(self, structure: Structure) -> dict:
		"""
		Check if structure has a rare space group.

		Returns:
			dict with space_group, sg_count_in_train, sg_rarity_score
		"""
		try:
			sg = SpacegroupAnalyzer(structure, symprec=0.1).get_space_group_number()
		except Exception:
			sg = 0

		sg_count = self.sg_counter.get(sg, 0)
		total = len(self.training_structures)

		# Rarity score: 1 - (count / total), higher = more novel
		rarity_score = 1.0 - (sg_count / total) if total > 0 else 1.0

		return {
			"space_group": sg,
			"sg_count_in_train": sg_count,
			"sg_rarity_score": rarity_score,
		}

	def structure_match_novelty(self, structure: Structure) -> dict:
		"""
		Check if structure matches any training structure.

		Returns:
			dict with has_match, match_index (or None)
		"""
		has_match = False
		match_index = None

		# Check against all training structures (expensive!)
		for i, train_struct in enumerate(self.training_structures):
			try:
				if self.matcher.fit(structure, train_struct):
					has_match = True
					match_index = i
					break
			except Exception:
				continue

		return {
			"has_match": has_match,
			"match_index": match_index,
			"structure_novel": not has_match,
		}

	def coordination_novelty(self, structure: Structure) -> dict:
		"""
		Check if coordination pattern is rare.

		Returns:
			dict with cn_pattern, cn_count_in_train, cn_rarity_score
		"""
		cn_pattern = self._get_cn_pattern(structure)
		cn_count = self.cn_counter.get(cn_pattern, 0)
		total = len(self.training_structures)

		# Rarity score
		rarity_score = 1.0 - (cn_count / total) if total > 0 else 1.0

		return {
			"cn_pattern": cn_pattern,
			"cn_count_in_train": cn_count,
			"cn_rarity_score": rarity_score,
		}

	def compute_novelty(
		self, structure: Structure, check_structure_match: bool = False
	) -> dict:
		"""
		Compute combined novelty score for a structure.

		Args:
			structure: Structure to analyze
			check_structure_match: If True, check StructureMatcher (slow!)

		Returns:
			dict with all novelty metrics and combined score
		"""
		result = {}

		# Space group novelty
		sg_result = self.space_group_novelty(structure)
		result.update(sg_result)

		# Coordination novelty
		cn_result = self.coordination_novelty(structure)
		result.update(cn_result)

		# Structure match novelty (optional, very slow)
		if check_structure_match:
			match_result = self.structure_match_novelty(structure)
			result.update(match_result)
			structure_novel_score = 1.0 if match_result["structure_novel"] else 0.0
		else:
			result["has_match"] = None
			result["match_index"] = None
			result["structure_novel"] = None
			structure_novel_score = 0.5  # neutral when not computed

		# Combined score (weighted average)
		# Higher = more novel
		combined = (
			sg_result["sg_rarity_score"] * 0.4
			+ cn_result["cn_rarity_score"] * 0.4
			+ structure_novel_score * 0.2
		)
		result["novelty_score"] = combined

		return result


def load_training_data() -> list[Structure]:
	"""Load MP-20 training structures."""
	print("Downloading MP-20 training data...")
	url = "https://raw.githubusercontent.com/txie-93/cdvae/refs/heads/main/data/mp_20/train.csv"
	response = requests.get(url, timeout=60)
	train_df = pd.read_csv(io.StringIO(response.content.decode("utf-8")))

	print(f"Parsing {len(train_df)} CIF strings...")
	structures = []
	for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Parsing CIFs"):
		try:
			struct = CifParser.from_str(row["cif"]).parse_structures(primitive=True)[0]
			structures.append(struct)
		except Exception:
			continue

	print(f"Loaded {len(structures)} training structures")
	return structures


def load_generated_samples(model: str = "chemeleon2") -> list[Structure]:
	"""Load generated samples from HuggingFace."""
	print(f"Downloading {model} generated samples...")
	path = hf_hub_download(
		repo_id="masahiro-negishi/xtalmet",
		filename=f"mp20/model/{model}.pkl.gz",
		repo_type="dataset",
		revision=HF_VERSION,
	)
	with gzip.open(path, "rb") as f:
		samples = pickle.load(f)

	print(f"Loaded {len(samples)} generated samples")
	return samples


def main():
	"""Main entry point."""
	# Load data
	train_structures = load_training_data()
	gen_samples = load_generated_samples("chemeleon2")

	# Initialize calculator
	calc = StructuralNoveltyCalculator(train_structures)

	# Compute novelty for all generated samples
	print("\nComputing structural novelty for generated samples...")
	results = []

	for idx, structure in enumerate(tqdm(gen_samples, desc="Computing novelty")):
		try:
			# Don't check structure match (too slow for 10k samples)
			novelty = calc.compute_novelty(structure, check_structure_match=False)
			novelty["index"] = idx
			novelty["composition"] = str(structure.composition.reduced_formula)
			results.append(novelty)
		except Exception as e:
			results.append(
				{
					"index": idx,
					"composition": "FAILED",
					"novelty_score": np.nan,
					"error": str(e),
				}
			)

	# Convert to array for statistics
	novelty_scores = np.array([r.get("novelty_score", np.nan) for r in results])

	# Save results
	output_path = "chemeleon2_structural_novelty.pkl.gz"
	with gzip.open(output_path, "wb") as f:
		pickle.dump(results, f)
	print(f"\nSaved results to {output_path}")

	# Print statistics
	print("\n" + "=" * 60)
	print("STRUCTURAL NOVELTY STATISTICS")
	print("=" * 60)
	print(f"Total samples: {len(results)}")
	print(f"Valid samples: {np.sum(~np.isnan(novelty_scores))}")
	print(f"\nNovelty Score:")
	print(f"  Mean:   {np.nanmean(novelty_scores):.4f}")
	print(f"  Std:    {np.nanstd(novelty_scores):.4f}")
	print(f"  Min:    {np.nanmin(novelty_scores):.4f}")
	print(f"  Max:    {np.nanmax(novelty_scores):.4f}")

	# Top novel structures
	print("\n" + "-" * 60)
	print("TOP 20 NOVEL STRUCTURES (by novelty score)")
	print("-" * 60)

	# Sort by novelty score
	sorted_results = sorted(
		[r for r in results if not np.isnan(r.get("novelty_score", np.nan))],
		key=lambda x: x["novelty_score"],
		reverse=True,
	)

	for i, r in enumerate(sorted_results[:20]):
		print(
			f"{i + 1:2d}. idx={r['index']:5d} | {r['composition']:15s} | "
			f"score={r['novelty_score']:.3f} | "
			f"SG={r.get('space_group', '?'):3} (n={r.get('sg_count_in_train', '?')}) | "
			f"CN rarity={r.get('cn_rarity_score', 0):.3f}"
		)

	# Space group distribution
	print("\n" + "-" * 60)
	print("RARE SPACE GROUPS IN GENERATED SAMPLES")
	print("-" * 60)

	sg_counts = Counter([r.get("space_group", 0) for r in results if "space_group" in r])
	rare_sgs = [
		(sg, count)
		for sg, count in sg_counts.most_common()
		if calc.sg_counter.get(sg, 0) <= 5
	]

	print(f"Space groups with <=5 occurrences in training set:")
	for sg, gen_count in rare_sgs[:15]:
		train_count = calc.sg_counter.get(sg, 0)
		print(f"  SG {sg:3d}: {gen_count:4d} generated, {train_count:3d} in training")


if __name__ == "__main__":
	main()
