"""Training script for Reinforcement Learning fine-tuning."""

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from src.utils.instantiators import instantiate_callbacks, instantiate_loggers


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_rl.yaml")
def main(cfg: DictConfig) -> None:
    print(f"Running with config: {OmegaConf.to_yaml(cfg)}")

    # Set up random seed
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Set up DataModule
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Set up Model
    model: LightningModule = hydra.utils.instantiate(cfg.rl_module)

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
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))


if __name__ == "__main__":
    main()
