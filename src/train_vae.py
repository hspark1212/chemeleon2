"""Training script for Variational Autoencoder."""

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, Trainer
from lightning.pytorch.loggers import Logger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from src.utils.checkpoint import resolve_checkpoint_path
from src.utils.instantiators import instantiate_callbacks, instantiate_loggers
from src.vae_module.vae_module import VAEModule


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_vae.yaml")
def main(cfg: DictConfig) -> None:
    OmegaConf.register_new_resolver(
        "hub", lambda name: str(resolve_checkpoint_path(name)), replace=True
    )

    print(f"Running with config: {OmegaConf.to_yaml(cfg)}")

    # Set up random seed
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Set up DataModule
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Set up Model
    if cfg.get("ckpt_path"):
        ckpt_path = str(resolve_checkpoint_path(cfg.ckpt_path))
        print(f"Loading model from checkpoint: {ckpt_path}")
        model: VAEModule = VAEModule.load_from_checkpoint(ckpt_path, weights_only=False)
    else:
        model: VAEModule = hydra.utils.instantiate(cfg.vae_module)

    # Set up Callbacks
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Set up Loggers
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))
    for lg in logger:
        if isinstance(lg, WandbLogger):
            full_cfg_container = OmegaConf.to_container(cfg, resolve=True)
            assert isinstance(full_cfg_container, dict)
            lg.log_hyperparams(full_cfg_container)  # type: ignore[arg-type]

    # Set up Trainer
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    # Train the model
    resume_from = cfg.get("resume_from")
    if resume_from:
        resume_from = str(resolve_checkpoint_path(resume_from))
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume_from)


if __name__ == "__main__":
    main()
