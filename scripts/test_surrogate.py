#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 06/11/2023
# Created for diffusion_nextsim
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from copy import deepcopy

# External modules
import hydra
from omegaconf import DictConfig, OmegaConf


# Internal modules


main_logger = logging.getLogger(__name__)


def test_task(cfg: DictConfig) -> None:
    # Import within main loop to speed up training on jean zay
    import wandb
    from hydra.utils import instantiate
    import torch
    from torch.utils.data import DataLoader
    import lightning.pytorch as pl

    from diffusion_nextsim.data import TrajectoryDataset
    from diffusion_nextsim.data.augmentation import Augmentation

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.epilogue_fusion = False
    torch._inductor.config.coordinate_descent_check_all_directions = True

    main_logger.info(f"Instantiating dataset <{cfg.dataset_path}>")
    dataset = TrajectoryDataset(
        cfg.dataset_path,
        aux_path=cfg.data.aux_path,
        delta_t=cfg.delta_t,
        n_cycles=cfg.n_cycles+cfg.data.n_input_steps,
        state_variables=cfg.data.state_variables,
        forcing_variables=cfg.data.forcing_variables,
        augmentation=Augmentation(),
        zip_path=cfg.data.zip_path,
    )
    data_loader = DataLoader(
        dataset, batch_size=cfg.data.batch_size,
        pin_memory=torch.cuda.is_available(),
        num_workers=cfg.data.n_workers
    )

    main_logger.info(f"Instantiating model <{cfg.surrogate._target_}>")
    model: pl.LightningModule = instantiate(cfg.surrogate)
    new_encoder = deepcopy(model.encoder)
    new_decoder = deepcopy(model.decoder)

    state_dict = torch.load(cfg.ckpt_path, map_location="cpu")["state_dict"]
    returned_load = model.load_state_dict(state_dict, strict=False)
    main_logger.info(f"Loaded checkpoint with: {returned_load}")

    if cfg.keep_encoder:
        model.encoder = new_encoder
    if cfg.keep_decoder:
        model.decoder = new_decoder


    training_logger = None
    if OmegaConf.select(cfg, "logger") is not None:
        training_logger = instantiate(cfg.logger)

    if OmegaConf.select(cfg, "callbacks") is not None:
        callbacks = []
        for _, callback_cfg in cfg.callbacks.items():
            if "_target_" in callback_cfg.keys():
                curr_callback = instantiate(callback_cfg)
                callbacks.append(curr_callback)
    else:
        callbacks = None

    main_logger.info(f"Instantiating trainer")
    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=training_logger
    )

    main_logger.info(f"Starting testing")
    trainer.test(model=model, dataloaders=data_loader)
    main_logger.info(f"Testing finished")
    wandb.finish()


@hydra.main(
    version_base=None, config_path='../configs/',
    config_name='surrogate_test'
)
def main_test(cfg: DictConfig) -> None:
    try:
        test_task(cfg)
    except MemoryError:
        pass


if __name__ == '__main__':
    main_test()
