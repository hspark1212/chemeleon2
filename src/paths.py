"""Default checkpoint paths - downloads from HF Hub."""

from src.utils.checkpoint_downloader import get_checkpoint

# Default checkpoints - always from HF Hub
DEFAULT_VAE_CKPT_PATH = get_checkpoint("alex_mp_20_vae")
DEFAULT_LDM_CKPT_PATH = get_checkpoint("alex_mp_20_ldm")
