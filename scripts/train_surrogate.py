#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 06/10/2023
# Created for diffusion_nextsim
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging

# External modules
import hydra
from omegaconf import DictConfig


# Internal modules
from diffusion_nextsim.train_utils import train_task


main_logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path='../configs/', config_name='surrogate'
)
def main_train(cfg: DictConfig) -> None:
    try:
        train_task(cfg, network_name="surrogate")
    except MemoryError:
        pass


if __name__ == '__main__':
    main_train()

