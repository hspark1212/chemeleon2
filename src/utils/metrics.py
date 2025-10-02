import os
import tempfile
import shutil
import warnings
from collections import defaultdict
from pathlib import Path
import gzip
import pickle
from dataclasses import dataclass, field


from tqdm import tqdm
import numpy as np
import pandas as pd
import amd
from scipy.linalg import sqrtm
from ase.calculators.calculator import Calculator
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram, PhaseDiagram, PDEntry
from pymatgen.core import Structure
from monty.serialization import loadfn


import torch
from src.utils.featurizer import featurize
from src.utils.cl_score import compute_clscore


BENCHMARK_DIR = Path(__file__).resolve().parent.parent.parent / "benchmarks"
PATH_REFERENCE_STRUCTURES = {
    "mp-20": BENCHMARK_DIR / "assets" / "mp_20_all_structure.json.gz",
    "mp-all": BENCHMARK_DIR / "assets" / "mp_all_unique_structure_250416.json.gz",
    "alex-mp-20": BENCHMARK_DIR / "assets" / "alex_mp_20_all_structure.json.gz",
}
PATH_REFERENCE_STRUCTURE_FEATURES = {
    "mp": BENCHMARK_DIR / "assets" / "mp_20_all_structure_features.pt",
    "alex": BENCHMARK_DIR / "assets" / "alex_mp_20_all_structure_features.pt",
}
PATH_REFERENCE_COMPOSITION_FEATURES = {
    "mp": BENCHMARK_DIR / "assets" / "mp_20_all_composition_features.pt",
    "alex": BENCHMARK_DIR / "assets" / "alex_mp_20_all_composition_features.pt",
}
PATH_PHASE_DIAGRAM = {  # find details in "benchmarks/assets/README.md"
    "mp-all": BENCHMARK_DIR / "assets" / "ppd-mp_all_entries_uncorrected_250409.pkl.gz",
    "alex-mp-20": BENCHMARK_DIR
    / "assets"
    / "ppd-alex_mp_20_entries_uncorrected_250730.pkl.gz",
}
###############################################################################
#                             Registered decorators                           #
###############################################################################
_REGISTRY: dict[str, callable] = {}


def register_metric(name: str):
    def _wrap(func):
        _REGISTRY[name] = func
        return func

    return _wrap


###############################################################################
#                             Metric functions                                #
###############################################################################


@register_metric("unique")
def _unique(ctx):
    groups = ctx.sm.group_structures(ctx.gen_structures)
    first_ids = {id(g[0]) for g in groups}
    return np.array([id(s) in first_ids for s in ctx.gen_structures])


@register_metric("novel")
def _novel(ctx):
    # Check novelty for each generated structure
    is_novel = []
    structures_iter = (
        tqdm(ctx.gen_structures, desc="Checking novelty")
        if ctx.progress_bar
        else ctx.gen_structures
    )
    for gen_structure in structures_iter:
        matching_refs = ctx._ref_structures_by_formula.get(
            gen_structure.reduced_formula, []
        )
        novel = True
        for ref_st in matching_refs:
            if ctx.sm.fit(ref_st, gen_structure):
                novel = False
                break
        is_novel.append(novel)
    return np.array(is_novel)


@register_metric("e_above_hull")
def _e_above_hull(ctx):
    e_above_hull_list = []
    structures_iter = (
        tqdm(ctx.gen_structures, desc="Calculating energy above hull")
        if ctx.progress_bar
        else ctx.gen_structures
    )
    for gen_structure in structures_iter:
        try:
            energy_above_hull = calculate_energy_above_hull(
                phase_diagram=ctx._pd,
                calc=ctx._calc,
                gen_structure=gen_structure,
            )
            e_above_hull_list.append(energy_above_hull)
        except Exception as e:
            print(
                f"Error calculating energy above hull for {gen_structure.composition}: {e}"
            )
            e_above_hull_list.append(np.nan)
    e_above_hull_list = np.array(e_above_hull_list)
    ctx._results["is_metastable"] = e_above_hull_list <= ctx.metastable_threshold
    ctx._results["is_stable"] = e_above_hull_list <= 0.0
    return e_above_hull_list


@register_metric("composition_validity")
def _composition_validity(ctx):
    return np.array([ctx._smact_validity_fn(s.composition) for s in ctx.gen_structures])


@register_metric("structure_diversity")
def _structure_diversity(ctx):
    gen_features = featurize(ctx.gen_structures, device="cpu")["structure_features"]
    ref_features = ctx._reference_structure_features
    fmd_score = frechet_distance(
        gen_embeddings=gen_features, ref_embeddings=ref_features
    )
    inv_fmd_score = 1 / (1 + fmd_score)  # Inverse FMD for interpretability
    return inv_fmd_score


@register_metric("composition_diversity")
def _composition_diversity(ctx):
    gen_features = featurize(ctx.gen_structures, device="cpu")["composition_features"]
    ref_features = ctx._reference_composition_features
    fmd_score = frechet_distance(
        gen_embeddings=gen_features, ref_embeddings=ref_features
    )
    inv_fmd_score = 1 / (1 + fmd_score)  # Inverse FMD for interpretability
    return inv_fmd_score


@register_metric("synthesizability")
def _synthesizability(ctx):
    clscore = compute_clscore(ctx.gen_structures, batch_size=200, use_cuda=ctx.use_cuda)
    return clscore > 0.5


###############################################################################
#                                 Metrics                                     #
###############################################################################
@dataclass
class Metrics:
    """Container for metrics to evaluate generated crystal structures.

    1. Unique: Identifies structures that are not duplicates within the generated set
    2. Novel: Identifies structures not found in the reference dataset
    3. E Above Hull: Calculates the energy above hull for each structure (Metastable/Stability)
    4. Composition Validity: Checks if the composition is valid using SMACT
    5. Structure Diversity: Computes inverse Fréchet distance (1/(1+FMD)) between generated and reference structure embeddings from VAE (higher is better)
    6. Composition Diversity: Computes inverse Fréchet distance (1/(1+FMD)) between generated and reference composition embeddings from VAE (higher is better)

    Parameters
    ----------
    metrics : list[str], optional
        List of metric names to compute. If None, uses default metrics.
        Default is None, which uses ["unique", "novel", "e_above_hull", "composition_validity", "structure_diversity", "composition_diversity"].
    reference_dataset : str, default="mp-20"
        Name of the reference dataset to use (options: "mp-20", "mp-all", "alex-mp-20").
    phase_diagram : str, default="mp-all" (options: "mp-all", "alex-mp-20")
        Name of the phase diagram to use for stability calculations
    sm : StructureMatcher, optional
        StructureMatcher instance for structural comparison. If None, a default one is created.
    metastable_threshold : float, default=0.1
        Threshold for metastability in e_above_hull calculations
    progress_bar : bool, default=True
        Whether to show a progress bar during metric calculations
    use_cuda : bool, default=True
        Whether to use CUDA for computations. If False, uses CPU.

    Attributes
    ----------
    reference_structures : list[Structure]
        List of reference structures loaded from the specified dataset
    _pd : PatchedPhaseDiagram or PhaseDiagram
        Phase diagram used for energy above hull calculations
    _calc : Calculator
        Calculator instance for energy calculations
    _smact_validity_fn : callable
        Function to check composition validity using SMACT
    _reference_structure_features : torch.Tensor
        Tensor of reference structure features for diversity calculations
    _reference_composition_features : torch.Tensor
        Tensor of reference composition features for diversity calculations
    _results : dict[str, list]
        Dictionary to store results of computed metrics


    Methods
    -------
    register_metric(name, func=None)
        Register a metric for calculation
    compute_metrics()
        Compute all registered metrics
    to_dataframe()
        Convert results to a pandas DataFrame
    to_csv(path)
        Save results to a CSV file

    Example
    -------
    >>> from pymatgen.core import Structure
    >>> # Assume we have a list of generated structures
    >>> gen_structures = [Structure(...), Structure(...)]
    >>>
    >>> # Create metrics object
    >>> m = Metrics(
    ...     reference_dataset="mp-20",
    ...     phase_diagram="mp-all",
    ...     sm=None,
    ...     calc=None,
    ... )
    >>> # Compute metrics
    >>> results = m.compute(gen_structures=gen_structures)
    >>> # Convert results to DataFrame
    >>> df = m.to_dataframe()
    """

    metrics: list[str] = None
    reference_dataset: str = "mp-20"
    phase_diagram: str = "mp-all"
    sm: StructureMatcher = None
    metastable_threshold: float = 0.1
    progress_bar: bool = True
    use_cuda: bool = True

    # runtime
    gen_structures: list[Structure] = field(default_factory=list, init=False)
    _reference_structures: list[Structure] = field(default_factory=list, init=False)
    _pd: PatchedPhaseDiagram | PhaseDiagram = field(default=None, init=False)
    _calc: Calculator = field(default=None, init=False)
    _smact_validity_fn: callable = field(default=None, init=False)
    _reference_structure_features: torch.Tensor = field(default=None, init=False)
    _reference_composition_features: torch.Tensor = field(default=None, init=False)
    _results: dict[str, list] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                "unique",
                "novel",
                "e_above_hull",
                "composition_validity",
                "structure_diversity",
                "composition_diversity",
            ]
            print(f"Using default metrics: {self.metrics}")

        # sm: `unique` and `novel`
        if ("unique" in self.metrics or "novel" in self.metrics) and self.sm is None:
            self.sm = StructureMatcher()

        # _reference_structures: `novel`
        if "novel" in self.metrics and not self._reference_structures:
            path_reference_dataset = PATH_REFERENCE_STRUCTURES[self.reference_dataset]
            self._reference_structures = loadfn(path_reference_dataset)
            print(f"Loaded reference {len(self._reference_structures)} structures")
            self._ref_structures_by_formula = defaultdict(list)
            for ref_structure in self._reference_structures:
                self._ref_structures_by_formula[ref_structure.reduced_formula].append(
                    ref_structure
                )

        # _pd: `e_above_hull`, `stable`
        if "e_above_hull" in self.metrics and self._pd is None:
            path_phase_diagram = PATH_PHASE_DIAGRAM[self.phase_diagram]
            with gzip.open(path_phase_diagram, "rb") as f:
                self._pd = pickle.load(f)
            print(
                f"Loaded phase diagram from {path_phase_diagram} with {len(self._pd)} entries"
            )

        # _calc: `e_above_hull`, `stable`
        if "e_above_hull" in self.metrics and self._calc is None:
            from mace.calculators import mace_mp

            self._calc = mace_mp(
                model="medium-mpa-0", device="cuda" if self.use_cuda else "cpu"
            )

        # _smact_validity_fn: `composition_validity`
        if "composition_validity" in self.metrics and self._smact_validity_fn is None:
            from smact.screening import smact_validity

            def safe_smact_validity(comp):
                try:
                    return smact_validity(comp, oxidation_states_set="icsd24")
                except TypeError:
                    return False

            self._smact_validity_fn = safe_smact_validity

        # _reference_structure_features: `structure_diversity`
        if (
            "structure_diversity" in self.metrics
            and self._reference_structure_features is None
        ):
            path_reference_structure_features = self.reference_dataset.split("-")[0]
            self._reference_structure_features = torch.load(
                PATH_REFERENCE_STRUCTURE_FEATURES[path_reference_structure_features],
                map_location="cpu",
                weights_only=False,
            )

        # _reference_composition_features: `composition_diversity`
        if (
            "composition_diversity" in self.metrics
            and self._reference_composition_features is None
        ):
            path_reference_composition_features = self.reference_dataset.split("-")[0]
            self._reference_composition_features = torch.load(
                PATH_REFERENCE_COMPOSITION_FEATURES[
                    path_reference_composition_features
                ],
                map_location="cpu",
                weights_only=False,
            )

    def compute(self, gen_structures: list[Structure]):
        self._results = {}
        self.gen_structures = gen_structures
        names = self.metrics
        for name in names:
            if name not in _REGISTRY:
                raise KeyError(f"{name} not registered")
            self._results[name] = _REGISTRY[name](self)

        # Compute SUN / MSUN
        if all(m in self._results for m in ["unique", "novel", "is_metastable"]):
            self._results["MSUN"] = (
                self._results["unique"]
                & self._results["novel"]
                & self._results["is_metastable"]
            )

        if all(m in self._results for m in ["unique", "novel", "is_stable"]):
            self._results["SUN"] = (
                self._results["unique"]
                & self._results["novel"]
                & self._results["is_stable"]
            )

        # Compute average values
        for name, values in self._results.copy().items():
            if isinstance(values, np.ndarray):
                self._results[f"avg_{name}"] = np.nanmean(values)
            else:
                self._results[f"avg_{name}"] = values

        return self._results

    def to_dataframe(self, include_structure: bool = False):
        """Convert results to a pandas DataFrame."""
        df = pd.DataFrame(self._results)
        if include_structure:
            df["cif"] = [s.to(fmt="cif") for s in self._gen_structures]
        return df

    def to_csv(self, path: str):
        """Save results to a CSV file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()
        df.to_csv(p, index=False)
        print(f"Saved results to {p}.")


###############################################################################
#                                  Utils                                      #
###############################################################################


def uses_only_these_elements(entry, element_set):
    entry_els = set(str(el) for el in entry.composition.elements)
    return entry_els.issubset(element_set)


def calculate_energy_above_hull(
    phase_diagram: PatchedPhaseDiagram,
    calc: Calculator,
    gen_structure: Structure,
):
    # Get the energy of the generated structure
    gen_energy = calc.get_potential_energy(gen_structure.to_ase_atoms())

    # Check if energy is None
    if gen_energy is None:
        raise ValueError("Generated structure has no energy.")

    # Create a PDEntry for the generated structure
    gen_entry = PDEntry(composition=gen_structure.composition, energy=gen_energy)

    # Calculate energy above hull
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.filterwarnings(
            "ignore",
            message=r"No suitable PhaseDiagrams found.*",
            category=UserWarning,
        )
        energy_above_hull = phase_diagram.get_e_above_hull(
            gen_entry, allow_negative=True, check_stable=False
        )
    return energy_above_hull


def frechet_distance(
    gen_embeddings: torch.Tensor, ref_embeddings: torch.Tensor
) -> float:
    """
    Compute Fréchet distance between two sets of embedding vectors.

    Args:
        gen_embeddings: Reference embeddings tensor of shape (M, L)
        ref_embeddings: Generated embeddings tensor of shape (N, L)

    Returns:
        Fréchet distance as a scalar
    """
    # Convert to numpy for computation
    gen_emb = gen_embeddings.detach().cpu().numpy()
    ref_emb = ref_embeddings.detach().cpu().numpy()

    # Compute means
    mu_1 = np.mean(gen_emb, axis=0)
    mu_2 = np.mean(ref_emb, axis=0)

    # Compute covariance matrices
    sigma_1 = np.cov(gen_emb.T)
    sigma_2 = np.cov(ref_emb.T)

    # Compute mean squared difference
    mean_diff = np.sum((mu_1 - mu_2) ** 2)

    # Compute sqrt of product of covariances
    sqrt_product = sqrtm(sigma_1 @ sigma_2)

    # Handle numerical issues with complex numbers
    if np.iscomplexobj(sqrt_product):
        sqrt_product = sqrt_product.real

    # Compute Fréchet distance
    trace_term = np.trace(sigma_1 + sigma_2 - 2 * sqrt_product)
    frechet_dist = mean_diff + trace_term

    return frechet_dist


def structures_to_amd(structures, k=100, from_structure=False):
    if from_structure:
        # Use SpaceGroupAnalyzer from Structure -> slower speed
        psets = [amd.periodicset_from_pymatgen_structure(st) for st in structures]
        amds = [amd.AMD(pset, k) for pset in psets]
        return amds
    else:
        # Suppose structures as P1 -> faster speed
        temp_dir = tempfile.mkdtemp()
        try:
            paths = []
            for i, st in enumerate(structures):
                path = os.path.join(temp_dir, f"{i:06d}.cif")
                st.to(filename=path, fmt="cif")
                paths.append(path)

            amds = []
            for path in paths:
                pset = amd.CifReader(path).read()
                amds.append(amd.AMD(pset, k))
            return amds
        finally:
            shutil.rmtree(temp_dir)
