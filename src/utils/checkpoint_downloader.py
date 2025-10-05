"""Checkpoint download utility using Hugging Face Hub."""

import hashlib
import time
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

CHECKPOINT_CONFIG = (
    Path(__file__).parent.parent.parent / "checkpoints" / "checkpoints.yaml"
)  # Available checkpoints config file


def load_checkpoint_config() -> dict:
    """Load checkpoint configuration from YAML file."""
    with open(CHECKPOINT_CONFIG) as f:
        return yaml.safe_load(f)


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_checkpoint(name: str) -> Path:
    """Download checkpoint from Hugging Face Hub if not cached.

    Args:
        name: Checkpoint name from config (e.g., 'mp_20_vae').

    Returns:
        Path to downloaded checkpoint.
    """
    # Load config
    config = load_checkpoint_config()
    ckpt_info = config["checkpoints"][name]
    hf_repo = config["config"]["hf_repo"]
    cache_dir = config["config"]["cache_dir"]
    retry_attempts = config["config"].get("retry_attempts", 3)
    retry_delay = config["config"].get("retry_delay", 2)

    # Download with retry
    last_error = None
    for attempt in range(retry_attempts):
        try:
            path = hf_hub_download(
                repo_id=hf_repo,
                filename=ckpt_info["hf_path"],
                cache_dir=cache_dir,
            )

            # Verify checksum
            actual = calculate_sha256(Path(path))
            expected = ckpt_info["sha256"]
            if actual != expected:
                raise ValueError(f"Checksum mismatch: {actual} != {expected}")

            return Path(path)

        except Exception as e:
            last_error = e
            if attempt < retry_attempts - 1:
                print(
                    f"⚠ Download failed (attempt {attempt + 1}/{retry_attempts}): {e}"
                )
                print(f"  Retrying in {retry_delay}s...")
                time.sleep(retry_delay)

    # All retries failed
    raise RuntimeError(
        f"Failed to download '{name}' after {retry_attempts} attempts"
    ) from last_error
