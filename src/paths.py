"""Default checkpoint paths - downloads from HF Hub."""

from src.utils.checkpoint import get_checkpoint, load_checkpoint_config

config = load_checkpoint_config()
AVAIL_CHECKPOINTS = ", ".join(config["checkpoints"].keys())

# Default checkpoints
DEFAULT_VAE_CKPT_PATH = get_checkpoint("alex_mp_20_vae")
DEFAULT_LDM_CKPT_PATH = get_checkpoint("alex_mp_20_ldm_rl")
