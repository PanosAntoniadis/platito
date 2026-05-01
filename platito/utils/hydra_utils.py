from typing import List

import hydra
from lightning import Callback
from omegaconf import DictConfig


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks
