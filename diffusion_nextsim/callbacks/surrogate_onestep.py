#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 11/10/2023
# Created for diffusion_nextsim
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Any, Iterable

# External modules
import matplotlib
matplotlib.use('agg')

import torch
import lightning.pytorch as pl

# Internal modules


main_logger = logging.getLogger(__name__)


class SurrogateOneStepCallback(pl.Callback):
    def __init__(
            self,
            variables: Iterable[str],
            std: Iterable[float],
            n_input_steps: int = 1,
            every_n_step: int = 250
    ):
        super().__init__()
        self.variables = variables
        self.std = torch.tensor(std)
        self.n_input_steps = n_input_steps
        self.every_n_step = every_n_step

    def estimate_error(
            self,
            pl_module: "pl.LightningModule",
            batch: Any,
    ) -> None:
        prediction = pl_module(
            batch["state_traj"][:, :self.n_input_steps],
            batch["forcing_traj"][:, :self.n_input_steps+1],
            mesh=batch["mesh"], mask=batch["mask"]
        )
        target = batch["state_traj"][:, self.n_input_steps]
        mse = torch.mean(
            (prediction-target).pow(2), dim=(0, 2, 3)
        )
        mae = torch.mean(
            (prediction - target).abs(), dim=(0, 2, 3)
        )
        bias = torch.mean(
            prediction-target, dim=(0, 2, 3)
        )
        score_dict = {
            f'scores/mse_{name}': score
            for name, score in zip(self.variables, mse)
        }
        score_dict.update({
            f'scores/mae_{name}': score
            for name, score in zip(self.variables, mae)
        })
        score_dict.update({
            f'scores/bias_{name}': score
            for name, score in zip(self.variables, bias)
        })
        pl_module.log_dict(
            score_dict, batch_size=batch["state_traj"].size(0), sync_dist=True
        )
        self.std = self.std.to(mse)
        nrmse = (mse / self.std**2).mean().sqrt()
        pl_module.log("scores/nrmse", nrmse)

    def on_train_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule"
    ) -> None:
        pl_module.log("scores/nrmse", torch.inf)

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if (self.every_n_step >= 1 and
                trainer.global_step > 0 and
                trainer.global_step % self.every_n_step == 0):
            self.estimate_error(pl_module, batch)
