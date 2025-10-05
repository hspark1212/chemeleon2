"""Hyperparameter sweep script for VAE training."""

from functools import partial

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from src.train_vae import main as run_train_vae


def run_sweep(cfg: DictConfig) -> None:
    with wandb.init() as run:
        # Deepcopy the original config
        _cfg = cfg.copy()

        # Log the hyperparameters
        wandb_config = run.config
        print(f"Running with config: {wandb_config}")

        # Update the config with wandb hyperparameters
        for key, value in wandb_config.items():
            OmegaConf.update(_cfg, key, value)

        # Call the training function with the updated config
        run_train_vae(_cfg)


@hydra.main(version_base="1.3", config_path="../configs", config_name="sweep_vae.yaml")
def main(cfg: DictConfig) -> None:
    # Convert sweep config to container for wandb
    sweep_config = OmegaConf.to_container(cfg.sweep_config, resolve=True)
    assert isinstance(sweep_config, dict)

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="sweep-vae")

    # Start the sweep agent
    wandb.agent(sweep_id, function=partial(run_sweep, cfg=cfg), count=cfg.sweep_count)


if __name__ == "__main__":
    main()
