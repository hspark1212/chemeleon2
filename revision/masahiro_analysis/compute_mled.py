#!/usr/bin/env python3
"""Calculate M_LED (Metric for Local Environment Diversity) for crystal structures.

M_LED quantifies structural complexity through Shannon entropy of:
1. Polyhedral environments (coordination geometry)
2. Chemical environments (neighboring elements)

Higher entropy indicates greater local structural diversity.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matminer.featurizers.site.fingerprint import OPSiteFingerprint
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Structure
from scipy.stats import entropy

# Constants
VALID_ELEMENTS = [
	"H",
	"He",
	"Li",
	"Be",
	"B",
	"C",
	"N",
	"O",
	"F",
	"Ne",
	"Na",
	"Mg",
	"Al",
	"Si",
	"P",
	"S",
	"Cl",
	"Ar",
	"K",
	"Ca",
	"Sc",
	"Ti",
	"V",
	"Cr",
	"Mn",
	"Fe",
	"Co",
	"Ni",
	"Cu",
	"Zn",
	"Ga",
	"Ge",
	"As",
	"Se",
	"Br",
	"Kr",
	"Rb",
	"Sr",
	"Y",
	"Zr",
	"Nb",
	"Mo",
	"Tc",
	"Ru",
	"Rh",
	"Pd",
	"Ag",
	"Cd",
	"In",
	"Sn",
	"Sb",
	"I",
	"Xe",
	"Cs",
	"Ba",
	"La",
	"Ce",
	"Pr",
	"Nd",
	"Pm",
	"Sm",
	"Eu",
	"Gd",
	"Tb",
	"Dy",
	"Ho",
	"Er",
	"Tm",
	"Yb",
	"Lu",
	"Hf",
	"Ta",
	"W",
	"Re",
	"Os",
	"Ir",
	"Pt",
	"Au",
	"Hg",
	"Tl",
	"Pb",
	"Bi",
	"Po",
	"At",
	"Rn",
	"Fr",
	"Ra",
	"Ac",
	"Th",
	"Pa",
	"U",
	"Np",
	"Pu",
	"Am",
	"Cm",
	"Bk",
	"Cf",
	"Es",
	"Fm",
	"Md",
	"No",
	"Lr",
	"Rf",
	"Db",
	"Sg",
	"Bh",
	"Hs",
	"Mt",
	"Ds",
	"Rg",
][:85]  # Only first 85 elements

N_POLYHEDRA = 37  # Number of polyhedral types from OPSiteFingerprint
POSITIONAL_ENCODING_DIM = 2 * N_POLYHEDRA + len(VALID_ELEMENTS)
POSITIONAL_SIGMA = 10.0


class MLEDCalculationError(Exception):
	"""Raised when M_LED calculation fails for a structure."""

	pass


class LocalEnvironmentDiversityCalculator:
	"""Calculator for M_LED (Metric for Local Environment Diversity).

	Attributes:
	    featurizer: Site fingerprint featurizer for polyhedral identification
	    crystal_nn: Nearest neighbor finder for chemical environment
	    elements: List of valid elements to consider
	"""

	def __init__(self, elements: list[str] = VALID_ELEMENTS, use_voronoi: bool = False):
		"""Initialize M_LED calculator.

		Args:
		    elements: List of valid element symbols to consider
		    use_voronoi: If True, use VoronoiFingerprint instead of OPSiteFingerprint
		"""
		self.elements = elements

		# Initialize featurizer for polyhedral identification
		if use_voronoi:
			from matminer.featurizers.site.fingerprint import VoronoiFingerprint

			self.featurizer = VoronoiFingerprint()
		else:
			self.featurizer = OPSiteFingerprint()

		# Initialize geometric nearest neighbor finder
		self.crystal_nn = CrystalNN(
			distance_cutoffs=None,
			x_diff_weight=0,
			porous_adjustment=False,
			search_cutoff=12,
		)

	@staticmethod
	def calculate_shannon_entropy(vector: np.ndarray) -> float:
		"""Calculate Shannon entropy of a probability distribution.

		Args:
		    vector: Count or frequency vector

		Returns:
		    Shannon entropy in bits
		"""
		# Normalize to probability distribution
		prob_dist = vector / np.sum(vector)

		# Clip to avoid log(0)
		prob_dist = np.clip(prob_dist, 1e-10, 1.0)

		return entropy(prob_dist, base=2)

	def positional_encoding(
		self, feature_vector: np.ndarray, sigma: float = POSITIONAL_SIGMA
	) -> np.ndarray:
		"""Apply Gaussian positional encoding to feature vector.

		This smooths the discrete feature counts into a continuous representation
		that captures local correlations between features.

		Args:
		    feature_vector: Feature count vector
		    sigma: Width of Gaussian kernel

		Returns:
		    Positionally encoded feature vector
		"""
		dim = len(feature_vector)
		nonzero_indices = np.nonzero(feature_vector)[0]

		if len(nonzero_indices) == 0:
			return np.zeros(dim)

		# Create Gaussian encoding for each nonzero feature
		encoded = np.zeros((len(nonzero_indices), dim))
		for i, idx in enumerate(nonzero_indices):
			# Gaussian centered at feature index
			distances = (np.arange(dim) - idx) ** 2
			encoded[i, :] = np.exp(-distances / (2 * sigma**2))

		# Average encodings
		return np.sum(encoded, axis=0) / len(nonzero_indices)

	def featurize_structure(
		self, structure: Structure
	) -> tuple[np.ndarray, float, float]:
		"""Extract local environment features and calculate entropy metrics.

		Args:
		    structure: Pymatgen Structure object

		Returns:
		    Tuple of (feature_vector, positional_entropy, polyhedral_entropy)

		Raises:
		    MLEDCalculationError: If featurization fails
		"""
		try:
			n_sites = len(structure)
			feature_dim = 2 * N_POLYHEDRA + len(self.elements)
			site_features = np.zeros((n_sites, feature_dim))

			for site_idx, site in enumerate(structure):
				# Get central atom symbol
				central_symbol = site.species.elements[0].symbol

				# Skip if element not in valid list
				if central_symbol not in self.elements:
					continue

				# 1. Polyhedral environment (coordination geometry)
				try:
					op_fingerprint = self.featurizer.featurize(structure, site_idx)
					polyhedra_type = np.argmax(op_fingerprint)
					site_features[site_idx, 2 * polyhedra_type] += 1
				except Exception as e:
					warnings.warn(
						f"Failed to compute polyhedra at site {site_idx}: {e}"
					)
					continue

				# 2. Chemical environment (neighboring elements)
				try:
					neighbor_dict = self.crystal_nn.get_cn_dict(structure, site_idx)
				except Exception:
					neighbor_dict = {}

				# Add central atom to chemical environment
				element_idx = self.elements.index(central_symbol)
				site_features[site_idx, 2 * N_POLYHEDRA + element_idx] += 1

				# Add neighboring elements
				for neighbor_symbol in neighbor_dict.keys():
					if neighbor_symbol in self.elements:
						neighbor_idx = self.elements.index(neighbor_symbol)
						site_features[site_idx, 2 * N_POLYHEDRA + neighbor_idx] += 1

			# Sum over all sites to get structure-level features
			structure_features = np.sum(site_features, axis=0)

			# Calculate entropy metrics
			# 1. Positional encoding entropy (captures correlations)
			positional_encoded = self.positional_encoding(structure_features)
			entropy_positional = self.calculate_shannon_entropy(positional_encoded)

			# 2. Polyhedral entropy (coordination geometry diversity)
			polyhedral_features = structure_features[: 2 * N_POLYHEDRA]
			entropy_polyhedral = self.calculate_shannon_entropy(polyhedral_features)

			return (
				structure_features,
				round(entropy_positional, 4),
				round(entropy_polyhedral, 4),
			)

		except Exception as e:
			raise MLEDCalculationError(f"Failed to featurize structure: {e}") from e

	def is_valid_structure(self, structure: Structure) -> bool:
		"""Check if structure contains only valid elements.

		Args:
		    structure: Pymatgen Structure object

		Returns:
		    True if structure is valid, False otherwise
		"""
		structure_elements = set(structure.composition.as_dict().keys())
		valid_elements_set = set(self.elements)

		# Check if there's any overlap with valid elements
		return not valid_elements_set.isdisjoint(structure_elements)

	def process_dataframe(
		self,
		df: pd.DataFrame,
		cif_column: str = "cif",
		sample_size: int | None = None,
	) -> pd.DataFrame:
		"""Calculate M_LED metrics for all structures in a DataFrame.

		Args:
		    df: Input DataFrame with CIF strings
		    cif_column: Name of column containing CIF data
		    sample_size: If provided, randomly sample this many structures

		Returns:
		    DataFrame with added M_LED columns

		Raises:
		    ValueError: If input validation fails
		"""
		if cif_column not in df.columns:
			raise ValueError(f"Column '{cif_column}' not found in DataFrame")

		if df.empty:
			raise ValueError("Input DataFrame is empty")

		df_result = df.copy()

		# Sample if requested
		if sample_size is not None and sample_size < len(df_result):
			print(f"Sampling {sample_size} structures from {len(df_result)}...")
			df_result = df_result.sample(n=sample_size, random_state=42)
			df_result = df_result.reset_index(drop=True)

		# Parse CIF strings to structures
		print(f"Parsing {len(df_result)} CIF strings...")
		structures = []
		failed_indices = []

		for idx, cif_string in enumerate(df_result[cif_column]):
			try:
				structure = Structure.from_str(cif_string, fmt="cif")
				structures.append(structure)
			except Exception as e:
				warnings.warn(f"Failed to parse structure at index {idx}: {e}")
				structures.append(None)
				failed_indices.append(idx)

		df_result["structure"] = structures

		# Filter valid structures
		print("Filtering valid structures...")
		valid_mask = [
			struct is not None and self.is_valid_structure(struct)
			for struct in structures
		]

		n_invalid = len(df_result) - sum(valid_mask)
		if n_invalid > 0:
			print(f"Removed {n_invalid} invalid structures")

		df_result = df_result[valid_mask].reset_index(drop=True)

		# Calculate M_LED metrics
		print(f"Calculating M_LED metrics for {len(df_result)} structures...")

		feature_vectors = []
		entropy_positional = np.zeros(len(df_result))
		entropy_polyhedral = np.zeros(len(df_result))

		for idx, structure in enumerate(df_result["structure"]):
			if idx % 100 == 0 and idx > 0:
				print(f"  Processed {idx}/{len(df_result)} structures...")

			try:
				features, ent_pos, ent_poly = self.featurize_structure(structure)
				feature_vectors.append(features)
				entropy_positional[idx] = ent_pos
				entropy_polyhedral[idx] = ent_poly
			except Exception as e:
				warnings.warn(f"Failed at index {idx}: {e}")
				feature_vectors.append(None)
				entropy_positional[idx] = np.nan
				entropy_polyhedral[idx] = np.nan

		# Add results to DataFrame
		df_result["mled_features"] = feature_vectors
		df_result["entropy_positional"] = entropy_positional
		df_result["entropy_polyhedral"] = entropy_polyhedral
		df_result["entropy_total"] = (
			df_result["entropy_positional"] + df_result["entropy_polyhedral"]
		)

		# Drop failed calculations
		df_result = df_result.dropna(subset=["entropy_total"])

		# Print statistics
		if len(df_result) > 0:
			print(f"\nM_LED statistics (n={len(df_result)}):")
			print("  Total entropy:")
			print(f"    Mean:   {df_result['entropy_total'].mean():.4f}")
			print(f"    Median: {df_result['entropy_total'].median():.4f}")
			print(f"    Std:    {df_result['entropy_total'].std():.4f}")
			print(
				f"    Range:  [{df_result['entropy_total'].min():.4f}, "
				f"{df_result['entropy_total'].max():.4f}]"
			)
			print(
				f"  Positional entropy: {df_result['entropy_positional'].mean():.4f} ± "
				f"{df_result['entropy_positional'].std():.4f}"
			)
			print(
				f"  Polyhedral entropy: {df_result['entropy_polyhedral'].mean():.4f} ± "
				f"{df_result['entropy_polyhedral'].std():.4f}"
			)

		return df_result


def main():
	"""Main entry point for the script."""
	parser = argparse.ArgumentParser(
		description="Calculate M_LED (Local Environment Diversity) metrics",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument(
		"input_file", type=str, help="Input CSV file containing structures"
	)
	parser.add_argument(
		"-o",
		"--output",
		type=str,
		default=None,
		help="Output CSV file (default: <input>_mled.csv)",
	)
	parser.add_argument(
		"--cif-column",
		type=str,
		default="cif",
		help="Name of column containing CIF strings",
	)
	parser.add_argument(
		"--sample",
		type=int,
		default=None,
		help="Randomly sample N structures (useful for large datasets)",
	)
	parser.add_argument(
		"--use-voronoi",
		action="store_true",
		help="Use Voronoi fingerprint instead of OP fingerprint",
	)
	parser.add_argument(
		"--drop-structure",
		action="store_true",
		help="Drop the structure column from output (saves space)",
	)

	args = parser.parse_args()

	# Validate input file
	input_path = Path(args.input_file)
	if not input_path.exists():
		print(f"Error: Input file '{args.input_file}' not found", file=sys.stderr)
		sys.exit(1)

	# Load data
	try:
		print(f"Loading data from {input_path}...")
		df = pd.read_csv(input_path, comment="#")
		print(f"Loaded {len(df)} rows")
	except Exception as e:
		print(f"Error loading CSV: {e}", file=sys.stderr)
		sys.exit(1)

	# Initialize calculator
	calculator = LocalEnvironmentDiversityCalculator(
		elements=VALID_ELEMENTS, use_voronoi=args.use_voronoi
	)

	# Process structures
	try:
		df_result = calculator.process_dataframe(
			df, cif_column=args.cif_column, sample_size=args.sample
		)
	except Exception as e:
		print(f"Error processing structures: {e}", file=sys.stderr)
		sys.exit(1)

	# Drop structure column if requested
	if args.drop_structure and "structure" in df_result.columns:
		df_result = df_result.drop(columns=["structure"])

	# Determine output filename
	if args.output:
		output_path = Path(args.output)
	else:
		stem = input_path.stem
		output_path = input_path.parent / f"{stem}_mled.csv"

	# Save results
	try:
		df_result.to_csv(output_path, index=False)
		print(f"\nResults saved to: {output_path}")
		print(
			"Added columns: entropy_positional, entropy_polyhedral, entropy_total, mled_features"
		)
	except Exception as e:
		print(f"Error saving output: {e}", file=sys.stderr)
		sys.exit(1)


if __name__ == "__main__":
	main()
