"""Evaluation script for generated crystal structures.

This script evaluates generated structures against reference datasets using
various metrics including structure matching, stability, and coverage.
"""

from pathlib import Path

import fire
from monty.serialization import loadfn
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from src.sample import sample
from src.utils.metrics import Metrics


#################
# main function #
#################
def evaluate(
    structure_path: str,  # Path to the generated structures (Directory or JSON file)
    model_path: str | None = None,  # Path to the trained model checkpoint
    reference_dataset: str = "mp-20",  # "mp-20", "mp-all"
    phase_diagram: str = "mp-all",  # "mp-all"
    output_file: str = "benchmark/results/benchmark_results.csv",
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
    stability_threshold: float = 0.1,
    **sample_configs,
) -> None:
    if model_path is not None:
        print(f"Using model from {model_path} for sampling structures.")
        sample(
            ldm_ckpt_path=model_path,
            output_dir=structure_path,
            **sample_configs,
        )

    if structure_path.endswith(".json") or structure_path.endswith(".gz"):
        gen_structures = loadfn(structure_path)
    else:
        structures_dir = Path(structure_path)
        gen_structures = [
            Structure.from_file(file) for file in structures_dir.glob("*.cif")
        ]
    print(f"Loaded {len(gen_structures)} generated structures from {structure_path}")

    # StructureMatcher parameters
    sm = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)

    m = Metrics(
        reference_dataset=reference_dataset,
        phase_diagram=phase_diagram,
        sm=sm,
        metastable_threshold=stability_threshold,
    )

    # Compute metrics
    m.compute(gen_structures=gen_structures)

    # Save results to CSV
    m.to_csv(output_file)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    fire.Fire(evaluate)
