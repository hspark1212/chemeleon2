"""Simple tests for M_LED calculator."""

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from compute_mled import LocalEnvironmentDiversityCalculator


@pytest.fixture
def nacl():
	"""Simple NaCl structure."""
	lattice = Lattice.cubic(5.64)
	return Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def test_calculator_init():
	"""Calculator initializes without error."""
	calc = LocalEnvironmentDiversityCalculator()
	assert calc is not None


def test_featurize_structure(nacl):
	"""Featurization runs and returns values."""
	calc = LocalEnvironmentDiversityCalculator()
	features, ent_pos, ent_poly = calc.featurize_structure(nacl)

	assert isinstance(features, np.ndarray)
	assert np.isfinite(ent_pos)
	assert np.isfinite(ent_poly)


def test_shannon_entropy():
	"""Shannon entropy computation works."""
	vector = np.array([1.0, 1.0, 1.0, 1.0])
	entropy = LocalEnvironmentDiversityCalculator.calculate_shannon_entropy(vector)
	assert np.isclose(entropy, 2.0, atol=1e-6)
