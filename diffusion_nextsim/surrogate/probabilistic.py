#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 09/11/2023
# Created for diffusion_nextsim
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Any, Dict

# External modules
import torch
import torch.nn
import lightning.pytorch as pl
from hydra.utils import instantiate
from omegaconf import OmegaConf

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

# Internal modules
from .utils import split_wd_params


main_logger = logging.getLogger(__name__)


class ProbabilisticModel(pl.LightningModule):
    def __init__(
            self,
            network: OmegaConf,
            encoder: OmegaConf,
            decoder: OmegaConf,
            lr: float = 1E-4,
            lr_warmup: int = 5000,
            total_steps: int = 250000,
            weight_decay: float = 1E-3
    ):
        super().__init__()
        self.network = instantiate(network)
        self.encoder = instantiate(encoder)
        self.decoder = instantiate(decoder)
        self.lr = lr
        self.lr_warmup = lr_warmup
        self.total_steps = total_steps
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def compile_model(self, compile: bool = True, use_ema: bool = True):
        if compile:
            self.network = torch.compile(self.network, mode="reduce-overhead")

    def forward(
            self,
            states: torch.Tensor,
            forcings: torch.Tensor,
            mesh: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        in_tensor = torch.cat((
            states.reshape(states.size(0), -1, *states.shape[-2:]),
            forcings.reshape(forcings.size(0), -1, *forcings.shape[-2:]),
        ), dim=-3)
        # Fixing the labels as no augmentation
        labels = torch.zeros(
            states.size(0), 3, dtype=states.dtype, device=states.device
        )
        encoded = self.encoder(in_tensor)
        out_tensor = self.network(
            encoded, labels=labels, mesh=mesh
        )
        return self.decoder(out_tensor, first_guess=states[:, -1], mask=mask)

    def estimate_loss(
            self,
            batch: Dict[str, torch.Tensor],
            prefix: str = "train",
    ) -> Dict[str, torch.Tensor]:
        # Check if scores should be synced
        sync_dist = prefix != "train"

        # Gather input and target data
        state_in = batch["state_traj"][:, :-1]
        forcing_in = batch["forcing_traj"]
        in_tensor = torch.cat((
            state_in.view(state_in.size(0), -1, *state_in.shape[-2:]),
            forcing_in.view(forcing_in.size(0), -1, *forcing_in.shape[-2:])
        ), dim=-3)

        encoded = self.encoder(in_tensor)
        prediction = self.network(
            encoded, labels=batch["labels"], mesh=batch["mesh"]
        )

        loss, _ = self.decoder.loss(
            in_tensor=prediction,
            first_guess=batch["state_traj"][:, -2],
            target=batch["state_traj"][:, -1],
            mask=batch["mask"]
        )
        self.log(
            f'{prefix}/loss', loss, batch_size=batch["state_traj"].size(0),
            prog_bar=True, sync_dist=sync_dist
        )
        return {"loss": loss, "prediction": prediction}

    def training_step(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int
    ) -> Any:
        total_loss = self.estimate_loss(batch, prefix="train")
        return total_loss

    def validation_step(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int,
    ) -> Any:
        loss = self.estimate_loss(batch, prefix="val")
        return loss

    def test_step(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int,
    ) -> Any:
        return None

    def configure_optimizers(
            self
    ) -> Any:
        wd_params, nowd_params = split_wd_params(self.network)
        optimizer = torch.optim.AdamW([
            {"params": wd_params, "weight_decay": self.weight_decay},
            {"params": nowd_params, "weight_decay": 0.0}
        ], lr=self.lr)
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer=optimizer,
            first_cycle_steps=self.total_steps,
            max_lr=self.lr,
            min_lr=1E-6,
            warmup_steps=self.lr_warmup,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
