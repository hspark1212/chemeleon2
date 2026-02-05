"""Benchmark sampling speed across Chemeleon2, Chemeleon1, and MatterGen.

This script measures the time to generate crystal structures using each framework
with GPU-synchronized timing for accurate measurements.

Usage:
    cd /home/hyunsoo/VScodeProjects/chemeleon2
    python revision/sampling_speed2/benchmark_sampling_speed.py
"""

import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

# Configuration paths
CHEMELEON2_PATH = Path("/home/hyunsoo/VScodeProjects/chemeleon2")
CHEMELEON1_PATH = Path("/home/hyunsoo/VScodeProjects/chemeleon-dng")
MATTERGEN_PATH = Path("/home/hyunsoo/VScodeProjects/mattergen")
OUTPUT_DIR = CHEMELEON2_PATH / "revision" / "sampling_speed2"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Checkpoint paths
CHEMELEON2_LDM_CKPT = (
    CHEMELEON2_PATH / "checkpoints/v0.0.1/mp_20/ldm/ldm_null_4tfw67aq.ckpt"
)
CHEMELEON2_VAE_CKPT = (
    CHEMELEON2_PATH / "checkpoints/v0.0.1/mp_20/vae/dng_m4owq4i5_v0.ckpt"
)

# Benchmark parameters
NUM_SAMPLES_LIST = [320, 1000]
BATCH_SIZE = 1000  # Single batch for all models (320 and 1000 fit in single batch)
NUM_WARMUP_RUNS = 1
NUM_TIMED_RUNS = 3  # for statistical significance


class GPUTimer:
    """GPU-synchronized timer for accurate CUDA timing.

    Uses CUDA synchronization and time.perf_counter() for high-resolution timing.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.use_cuda = torch.cuda.is_available() and device == "cuda"

    @contextmanager
    def time_block(self, description: str = ""):
        """Context manager for timing a block of code with GPU sync."""
        if self.use_cuda:
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        yield

        if self.use_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if description:
            print(f"  {description}: {elapsed:.2f}s")

        # Store the result for retrieval
        self._last_elapsed = elapsed

    def get_last_elapsed(self) -> float:
        """Get the elapsed time from the last time_block."""
        return getattr(self, "_last_elapsed", 0.0)

    def time_function(self, func: Callable, *args, **kwargs) -> tuple[any, float]:
        """Time a function call with GPU sync.

        Returns:
            Tuple of (function_result, elapsed_time)
        """
        if self.use_cuda:
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        result = func(*args, **kwargs)

        if self.use_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        return result, elapsed


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    model: str
    num_samples: int
    run_idx: int
    total_time_seconds: float
    samples_per_second: float
    per_batch_times: list[float] = field(default_factory=list)
    success: bool = True
    error_message: str | None = None


def clear_cuda_cache():
    """Clear CUDA cache between runs."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def setup_paths():
    """Add project paths to sys.path for imports."""
    for path in [CHEMELEON2_PATH, CHEMELEON1_PATH, MATTERGEN_PATH]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def get_num_atoms_distribution(num_samples: int, distribution: str = "mp-20") -> list[int]:
    """Get a list of num_atoms based on the specified distribution."""
    # Use Chemeleon2's distribution
    sys.path.insert(0, str(CHEMELEON2_PATH))
    from src.data.num_atom_distributions import NUM_ATOM_DISTRIBUTIONS

    dist = NUM_ATOM_DISTRIBUTIONS[distribution]
    num_atoms = np.random.choice(
        list(dist.keys()),
        p=list(dist.values()),
        size=num_samples,
    ).tolist()
    return num_atoms


class Chemeleon2Sampler:
    """Sampler for Chemeleon2 model."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.timer = GPUTimer(device)

    def load_model(self):
        """Load the Chemeleon2 LDM model."""
        if self.model is not None:
            return

        sys.path.insert(0, str(CHEMELEON2_PATH))
        from src.ldm_module.ldm_module import LDMModule

        print("Loading Chemeleon2 model...")
        self.model = LDMModule.load_from_checkpoint(
            str(CHEMELEON2_LDM_CKPT),
            vae_ckpt_path=str(CHEMELEON2_VAE_CKPT),
            map_location=self.device,
            weights_only=False,
        )
        self.model.eval()
        print("Chemeleon2 model loaded.")

    def sample(
        self,
        num_samples: int,
        batch_size: int = 1000,
        sampling_steps: int = 50,
        sampler: str = "ddim",
        progress: bool = True,
    ) -> tuple[list, float, list[float]]:
        """Sample crystal structures.

        Returns:
            Tuple of (structures, total_time, per_batch_times)
        """
        self.load_model()

        sys.path.insert(0, str(CHEMELEON2_PATH))
        from src.data.schema import create_empty_batch

        num_atoms = get_num_atoms_distribution(num_samples)
        batch_size = min(batch_size, num_samples)

        all_structures = []
        per_batch_times = []
        total_time = 0.0

        for i in range(0, num_samples, batch_size):
            batch_num_atoms = num_atoms[i:i + batch_size]
            batch = create_empty_batch(batch_num_atoms, device=self.device)

            # Time this batch
            def sample_batch():
                with torch.no_grad():
                    return self.model.sample(
                        batch,
                        sampler=sampler,
                        sampling_steps=sampling_steps,
                        return_structure=True,
                        progress=progress,
                    )

            structures, elapsed = self.timer.time_function(sample_batch)
            all_structures.extend(structures)
            per_batch_times.append(elapsed)
            total_time += elapsed

        return all_structures, total_time, per_batch_times


class Chemeleon1Sampler:
    """Sampler for Chemeleon1 (chemeleon-dng) model."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.timer = GPUTimer(device)

    def load_model(self):
        """Load the Chemeleon1 diffusion model."""
        if self.model is not None:
            return

        sys.path.insert(0, str(CHEMELEON1_PATH))
        from chemeleon_dng.diffusion.diffusion_module import DiffusionModule
        from chemeleon_dng.download_util import get_checkpoint_path

        print("Loading Chemeleon1 model...")
        default_model_path = {
            "dng": "ckpts/chemeleon_dng_alex_mp_20_v0.0.2.ckpt",
        }
        model_path = get_checkpoint_path("dng", default_model_path)

        self.model = DiffusionModule.load_from_checkpoint(
            model_path,
            map_location=self.device,
        )
        self.model.eval()
        print(f"Chemeleon1 model loaded (num_timesteps={self.model.num_timesteps}).")

    def sample(
        self,
        num_samples: int,
        batch_size: int = 1000,
        verbose: bool = True,
    ) -> tuple[list, float, list[float]]:
        """Sample crystal structures.

        Chemeleon1 uses 256 timesteps by default.

        Returns:
            Tuple of (structures, total_time, per_batch_times)
        """
        self.load_model()

        sys.path.insert(0, str(CHEMELEON1_PATH))
        from chemeleon_dng.dataset.num_atom_distributions import NUM_ATOM_DISTRIBUTIONS

        dist = NUM_ATOM_DISTRIBUTIONS["alex_mp_20"]
        num_atoms = np.random.choice(
            list(dist.keys()),
            p=list(dist.values()),
            size=num_samples,
        ).tolist()

        batch_size = min(batch_size, num_samples)

        all_atoms = []
        per_batch_times = []
        total_time = 0.0

        for i in range(0, num_samples, batch_size):
            batch_num_atoms = num_atoms[i:i + batch_size]

            def sample_batch():
                with torch.no_grad():
                    return self.model.sample(
                        task="dng",
                        num_atoms=batch_num_atoms,
                        verbose=verbose,
                    )

            atoms, elapsed = self.timer.time_function(sample_batch)
            all_atoms.extend(atoms)
            per_batch_times.append(elapsed)
            total_time += elapsed

        return all_atoms, total_time, per_batch_times


class MatterGenSampler:
    """Sampler for MatterGen model."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.generator = None
        self.timer = GPUTimer(device)

    def load_model(self):
        """Load the MatterGen generator."""
        if self.generator is not None:
            return

        sys.path.insert(0, str(MATTERGEN_PATH))
        from mattergen.generator import CrystalGenerator
        from mattergen.common.utils.data_classes import MatterGenCheckpointInfo

        print("Loading MatterGen model...")
        # Use the pretrained model from HuggingFace Hub
        checkpoint_info = MatterGenCheckpointInfo.from_hf_hub("mp_20_base")

        self.generator = CrystalGenerator(
            checkpoint_info=checkpoint_info,
            batch_size=1000,
            num_batches=1,
            record_trajectories=False,
        )
        self.generator.prepare()
        print("MatterGen model loaded (N=1000 timesteps).")

    def sample(
        self,
        num_samples: int,
        batch_size: int = 1000,
    ) -> tuple[list, float, list[float]]:
        """Sample crystal structures.

        MatterGen uses N=1000 timesteps.

        Returns:
            Tuple of (structures, total_time, per_batch_times)
        """
        self.load_model()

        batch_size = min(batch_size, num_samples)
        num_batches = max(1, (num_samples + batch_size - 1) // batch_size)
        actual_samples_per_batch = num_samples // num_batches

        all_structures = []
        per_batch_times = []
        total_time = 0.0

        for i in range(num_batches):
            # Adjust last batch if needed
            if i == num_batches - 1:
                current_batch_size = num_samples - len(all_structures)
            else:
                current_batch_size = actual_samples_per_batch

            def sample_batch():
                # Temporarily redirect stdout to suppress MatterGen's verbose output
                import io
                from contextlib import redirect_stdout

                f = io.StringIO()
                with redirect_stdout(f):
                    return self.generator.generate(
                        batch_size=current_batch_size,
                        num_batches=1,
                    )

            structures, elapsed = self.timer.time_function(sample_batch)
            all_structures.extend(structures)
            per_batch_times.append(elapsed)
            total_time += elapsed

        return all_structures, total_time, per_batch_times


def run_warmup(sampler, num_samples: int = 8):
    """Run warmup to trigger JIT compilation and load model."""
    print(f"  Running warmup with {num_samples} samples...")
    clear_cuda_cache()

    try:
        if isinstance(sampler, Chemeleon2Sampler):
            sampler.sample(num_samples, batch_size=num_samples, progress=False)
        elif isinstance(sampler, Chemeleon1Sampler):
            sampler.sample(num_samples, batch_size=num_samples, verbose=False)
        elif isinstance(sampler, MatterGenSampler):
            sampler.sample(num_samples, batch_size=num_samples)
    except Exception as e:
        print(f"  Warmup failed: {e}")


def benchmark_model(
    model_name: str,
    sampler,
    num_samples: int,
    batch_size: int,
    num_runs: int,
    do_warmup: bool = True,
) -> list[BenchmarkResult]:
    """Benchmark a single model with multiple runs."""
    results = []

    if do_warmup:
        for _ in range(NUM_WARMUP_RUNS):
            run_warmup(sampler)

    for run_idx in range(1, num_runs + 1):
        print(f"  Run {run_idx}/{num_runs} - {num_samples} samples...")
        clear_cuda_cache()

        try:
            if isinstance(sampler, Chemeleon2Sampler):
                _, total_time, per_batch_times = sampler.sample(
                    num_samples,
                    batch_size=batch_size,
                    sampling_steps=50,
                    sampler="ddim",
                    progress=False,
                )
            elif isinstance(sampler, Chemeleon1Sampler):
                _, total_time, per_batch_times = sampler.sample(
                    num_samples,
                    batch_size=batch_size,
                    verbose=False,
                )
            elif isinstance(sampler, MatterGenSampler):
                _, total_time, per_batch_times = sampler.sample(
                    num_samples,
                    batch_size=batch_size,
                )
            else:
                raise ValueError(f"Unknown sampler type: {type(sampler)}")

            samples_per_sec = num_samples / total_time if total_time > 0 else 0

            result = BenchmarkResult(
                model=model_name,
                num_samples=num_samples,
                run_idx=run_idx,
                total_time_seconds=total_time,
                samples_per_second=samples_per_sec,
                per_batch_times=per_batch_times,
                success=True,
            )
            print(f"    Time: {total_time:.2f}s, Speed: {samples_per_sec:.2f} samples/s")

        except Exception as e:
            result = BenchmarkResult(
                model=model_name,
                num_samples=num_samples,
                run_idx=run_idx,
                total_time_seconds=0.0,
                samples_per_second=0.0,
                per_batch_times=[],
                success=False,
                error_message=str(e),
            )
            print(f"    FAILED: {e}")

        results.append(result)

    return results


def run_all_benchmarks() -> list[BenchmarkResult]:
    """Run benchmarks for all models and sample counts."""
    all_results = []

    # Initialize samplers
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    models = {
        "Chemeleon2": Chemeleon2Sampler(device),
        "Chemeleon1": Chemeleon1Sampler(device),
        "MatterGen": MatterGenSampler(device),
    }

    for model_name, sampler in models.items():
        print(f"\n{'=' * 60}")
        print(f"Benchmarking {model_name}")
        print(f"{'=' * 60}")

        for num_samples in NUM_SAMPLES_LIST:
            print(f"\n{model_name} - {num_samples} samples:")
            results = benchmark_model(
                model_name=model_name,
                sampler=sampler,
                num_samples=num_samples,
                batch_size=BATCH_SIZE,
                num_runs=NUM_TIMED_RUNS,
                do_warmup=True,
            )
            all_results.extend(results)

    return all_results


def save_results(results: list[BenchmarkResult]) -> pd.DataFrame:
    """Save benchmark results to CSV and JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Create flattened data for CSV
    csv_data = []
    for r in results:
        csv_data.append({
            "model": r.model,
            "num_samples": r.num_samples,
            "run_idx": r.run_idx,
            "total_time_seconds": r.total_time_seconds,
            "samples_per_second": r.samples_per_second,
            "success": r.success,
            "error_message": r.error_message,
        })

    df = pd.DataFrame(csv_data)
    csv_path = RESULTS_DIR / "raw_timing_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Create complete data with per-batch times for JSON
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_samples_list": NUM_SAMPLES_LIST,
            "batch_size": BATCH_SIZE,
            "num_warmup_runs": NUM_WARMUP_RUNS,
            "num_timed_runs": NUM_TIMED_RUNS,
        },
        "results": [
            {
                "model": r.model,
                "num_samples": r.num_samples,
                "run_idx": r.run_idx,
                "total_time_seconds": r.total_time_seconds,
                "samples_per_second": r.samples_per_second,
                "per_batch_times": r.per_batch_times,
                "success": r.success,
                "error_message": r.error_message,
            }
            for r in results
        ],
    }

    json_path = RESULTS_DIR / "raw_timing_data.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Results saved to: {json_path}")

    return df


def create_visualizations(df: pd.DataFrame) -> None:
    """Create seaborn visualizations for benchmark results."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Filter successful runs only
    df_success = df[df["success"]].copy()

    if df_success.empty:
        print("No successful runs to visualize!")
        return

    # Set style
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 12

    # Color palette for models
    model_colors = {
        "Chemeleon2": "#2ecc71",  # Green
        "Chemeleon1": "#3498db",  # Blue
        "MatterGen": "#e74c3c",   # Red
    }

    # 1. Sampling Speed Comparison (Total Time with Error Bars)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, num_samples in enumerate(NUM_SAMPLES_LIST):
        ax = axes[idx]
        data = df_success[df_success["num_samples"] == num_samples]

        sns.barplot(
            data=data,
            x="model",
            y="total_time_seconds",
            hue="model",
            palette=model_colors,
            ax=ax,
            errorbar="sd",
            capsize=0.15,
            legend=False,
        )
        ax.set_title(
            f"Sampling Time ({num_samples} samples)",
            fontsize=14,
            fontweight="bold"
        )
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Time (seconds)", fontsize=12)

        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f", fontsize=10, padding=3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "sampling_speed_comparison.png", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "sampling_speed_comparison.pdf", bbox_inches="tight")
    print(f"Saved: {FIGURES_DIR / 'sampling_speed_comparison.png'}")
    plt.close(fig)

    # 2. Throughput Comparison (Samples per Second)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, num_samples in enumerate(NUM_SAMPLES_LIST):
        ax = axes[idx]
        data = df_success[df_success["num_samples"] == num_samples]

        sns.barplot(
            data=data,
            x="model",
            y="samples_per_second",
            hue="model",
            palette=model_colors,
            ax=ax,
            errorbar="sd",
            capsize=0.15,
            legend=False,
        )
        ax.set_title(
            f"Throughput ({num_samples} samples)",
            fontsize=14,
            fontweight="bold"
        )
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Samples per Second", fontsize=12)

        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=10, padding=3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "throughput_comparison.png", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "throughput_comparison.pdf", bbox_inches="tight")
    print(f"Saved: {FIGURES_DIR / 'throughput_comparison.png'}")
    plt.close(fig)

    # 3. Per-batch Timing Distribution (Box Plot)
    # Load JSON to get per-batch times
    json_path = RESULTS_DIR / "raw_timing_data.json"
    with open(json_path) as f:
        json_data = json.load(f)

    batch_data = []
    for r in json_data["results"]:
        if r["success"]:
            for batch_idx, batch_time in enumerate(r["per_batch_times"]):
                batch_data.append({
                    "model": r["model"],
                    "num_samples": r["num_samples"],
                    "run_idx": r["run_idx"],
                    "batch_idx": batch_idx,
                    "batch_time": batch_time,
                })

    if batch_data:
        df_batch = pd.DataFrame(batch_data)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Create grouped boxplot
        df_batch["config"] = df_batch["model"] + "\n(" + df_batch["num_samples"].astype(str) + " samples)"

        sns.boxplot(
            data=df_batch,
            x="config",
            y="batch_time",
            hue="model",
            palette=model_colors,
            ax=ax,
            legend=False,
        )
        ax.set_title("Per-Batch Timing Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Model (Samples)", fontsize=12)
        ax.set_ylabel("Batch Time (seconds)", fontsize=12)
        plt.xticks(rotation=15)

        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "per_batch_timing.png", bbox_inches="tight")
        fig.savefig(FIGURES_DIR / "per_batch_timing.pdf", bbox_inches="tight")
        print(f"Saved: {FIGURES_DIR / 'per_batch_timing.png'}")
        plt.close(fig)

    # 4. Summary Statistics Table
    summary = (
        df_success.groupby(["model", "num_samples"])
        .agg(
            mean_time=("total_time_seconds", "mean"),
            std_time=("total_time_seconds", "std"),
            mean_throughput=("samples_per_second", "mean"),
            std_throughput=("samples_per_second", "std"),
            n_runs=("total_time_seconds", "count"),
        )
        .round(3)
    )

    # Create table visualization
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    # Reset index for table display
    summary_reset = summary.reset_index()
    summary_reset.columns = [
        "Model", "Samples", "Mean Time (s)", "Std Time (s)",
        "Mean Throughput", "Std Throughput", "N Runs"
    ]

    table = ax.table(
        cellText=summary_reset.values,
        colLabels=summary_reset.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color header row
    for i in range(len(summary_reset.columns)):
        table[(0, i)].set_facecolor("#4a90d9")
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    # Color model rows
    for row_idx, row in enumerate(summary_reset.values):
        model = row[0]
        color = model_colors.get(model, "#ffffff")
        for col_idx in range(len(row)):
            table[(row_idx + 1, col_idx)].set_facecolor(color + "40")  # Add transparency

    plt.title("Benchmark Summary Statistics", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "summary_table.png", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "summary_table.pdf", bbox_inches="tight")
    print(f"Saved: {FIGURES_DIR / 'summary_table.png'}")
    plt.close(fig)

    # Print summary to console
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(summary.to_string())

    # Save summary CSV
    summary_path = RESULTS_DIR / "summary_statistics.csv"
    summary.to_csv(summary_path)
    print(f"\nSaved: {summary_path}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("SAMPLING SPEED BENCHMARK")
    print("=" * 60)
    print(f"Models: Chemeleon2, Chemeleon1, MatterGen")
    print(f"Sample counts: {NUM_SAMPLES_LIST}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Warmup runs: {NUM_WARMUP_RUNS}")
    print(f"Timed runs: {NUM_TIMED_RUNS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    # Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Setup paths
    setup_paths()

    # Run benchmarks
    results = run_all_benchmarks()

    # Save results
    df = save_results(results)

    # Create visualizations
    create_visualizations(df)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
