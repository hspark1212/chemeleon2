import warnings

import numpy as np
from matminer.featurizers.site.fingerprint import OPSiteFingerprint
from pymatgen.core import Structure
from scipy.stats import entropy

N_POLYHEDRA = 37
_FEATURIZER = None


def _get_featurizer():
	global _FEATURIZER
	if _FEATURIZER is None:
		_FEATURIZER = OPSiteFingerprint()
	return _FEATURIZER


def compute_structural_diversity(structure: Structure) -> float:
	featurizer = _get_featurizer()
	n_sites = len(structure)
	polyhedra_counts = np.zeros(N_POLYHEDRA)

	for site_idx in range(n_sites):
		try:
			op_fingerprint = featurizer.featurize(structure, site_idx)
			polyhedra_type = np.argmax(op_fingerprint)
			polyhedra_counts[polyhedra_type] += 1
		except Exception as e:
			warnings.warn(f"Failed at site {site_idx}: {e}")
			continue

	if polyhedra_counts.sum() == 0:
		return 0.0

	prob_dist = polyhedra_counts / polyhedra_counts.sum()
	prob_dist = np.clip(prob_dist, 1e-10, 1.0)
	return float(entropy(prob_dist, base=2))
