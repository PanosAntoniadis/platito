import wandb
import torch
import hydra
import lightning as L

from typing import List
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning_utilities.core.rank_zero import rank_zero_only
from lightning import LightningDataModule, LightningModule, Callback, Trainer

from platito.utils.hydra_utils import instantiate_callbacks


@rank_zero_only
def setup_logger(cfg: DictConfig):
    logger_config = OmegaConf.to_container(cfg, resolve=True)
    run = wandb.init(
        **cfg.logger,
        config=logger_config,
    )
    return WandbLogger(experiment=run)


@hydra.main(
    version_base=None, config_path="../configs", config_name="train.yaml"
)
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("medium")
    L.seed_everything(12345, workers=True)
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    model: LightningModule = hydra.utils.instantiate(cfg.model)
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    if cfg.get("logger"):
        logger = setup_logger(cfg)
    else:
        logger = None

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    if cfg.get("train"):
        trainer.fit(
            model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
        )

    if cfg.get("validate"):
        datamodule.setup("val")
        trainer.validate(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
        )


if __name__ == "__main__":
    main()
