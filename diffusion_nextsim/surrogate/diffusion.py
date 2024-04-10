#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 31/10/2023
# Created for diffusion_nextsim
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Dict, Any

# External modules
import torch
import lightning.pytorch as pl

from hydra.utils import instantiate
from omegaconf import OmegaConf

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

# Internal modules
from ddm_dynamical.utils import sample_time, normalize_gamma
from ddm_dynamical.scheduler import LinearScheduler
from ddm_dynamical.weighting import ELBOWeighting
from ddm_dynamical.parameterization import VParam
from .utils import split_wd_params


main_logger = logging.getLogger(__name__)


def residual_preprocessing(
        in_data: torch.Tensor,
        alpha: torch.Tensor,
        sigma: torch.Tensor,
        gamma: torch.Tensor,
        mask: torch.Tensor,
        encoded: torch.Tensor,
        **conditioning
):
    return torch.cat((in_data, encoded), dim=1)


class DiffusionModel(pl.LightningModule):
    def __init__(
            self,
            network: OmegaConf,
            encoder: OmegaConf,
            decoder: OmegaConf,
            scheduler: OmegaConf,
            sampler: OmegaConf,
            weighting: OmegaConf = None,
            param: OmegaConf = None,
            lr: float = 1E-4,
            lr_warmup: int = 5000,
            total_steps: int = 250000,
            weight_decay: float = 1E-3,
            ema_rate: float = 0.999,
            use_ema: bool = True,
    ):
        super().__init__()
        # Diffusion model
        self.network = instantiate(network)
        self.ema_model = torch.optim.swa_utils.AveragedModel(
            self.network,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                ema_rate
            )
        )
        self.ema_model.requires_grad_(False)

        # Scheduler, weighting and sampler for diffusion model
        self.scheduler = instantiate(scheduler)
        self.clim_scheduler = LinearScheduler(
            gamma_min=self.scheduler.gamma_min,
            gamma_max=self.scheduler.gamma_max
        )
        self.param = instantiate(param) or VParam()
        self.weighting = instantiate(weighting) or ELBOWeighting()
        self.sampler = instantiate(
            sampler,
            pre_func=residual_preprocessing,
            param=self.param,
            denoising_network=self.ema_model if use_ema else self.network
        )
        self.sampler.gamma_min = self.gamma_min = self.scheduler.gamma_min
        self.sampler.gamma_max = self.gamma_max = self.scheduler.gamma_max
        self.sampler.requires_grad_(False)

        # Encoder and decoder
        self.encoder = instantiate(encoder)
        self.decoder = instantiate(decoder)

        # Parameters
        self.ema_rate = ema_rate
        self.lr = lr
        self.lr_warmup = lr_warmup
        self.total_steps = total_steps
        self.weight_decay = weight_decay

        self.save_hyperparameters()

    def normalize_gamma(self, gamma: torch.Tensor):
        return (gamma - self.gamma_min) / (self.gamma_max - self.gamma_min)

    def compile_model(self, compile: bool = True, use_ema: bool = True):
        self.sampler.denoising_network = (
            self.ema_model if use_ema else self.network
        )
        if compile:
            self.sampler.denoising_network = torch.compile(
                self.sampler.denoising_network, mode="reduce-overhead"
            )

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
        dynamics = self.sampler.sample(
            states[:, -1].shape,
            encoded=encoded,
            labels=labels,
            mesh=mesh,
            mask=mask,
            decoder=self.decoder,
            first_guess=states[:, -1]
        )
        return self.decoder(
            dynamics,
            first_guess=states[:, -1],
            mask=mask
        )

    def estimate_loss(
            self,
            batch: Dict[str, torch.Tensor],
            prefix: str = "train"
    ) -> Dict[str, torch.Tensor]:
        # Check if scores should be synced
        sync_dist = prefix != "train"

        # Input data
        state_in = batch["state_traj"][:, :-1]
        forcing_in = batch["forcing_traj"]
        in_tensor = torch.cat((
            state_in.view(state_in.size(0), -1, *state_in.shape[-2:]),
            forcing_in.view(forcing_in.size(0), -1, *forcing_in.shape[-2:])
        ), dim=-3)

        residual = batch["state_traj"][:, -1] - batch["state_traj"][:, -2]
        residual = (residual - self.decoder.mean) / self.decoder.std

        ## Diffuse model
        noise = torch.randn_like(residual)
        sampled_time = sample_time(residual)
        gamma = self.scheduler(sampled_time)
        alpha_sq = torch.sigmoid(gamma)
        alpha = alpha_sq.sqrt()
        sigma = (1-alpha_sq).sqrt()
        noised_residual = alpha * residual + sigma * noise

        ## Estimate prediction with diffusion model
        encoded = self.encoder(in_tensor)
        in_tensor = torch.cat(
            (noised_residual, encoded), dim=1
        )
        normalized_gamma = normalize_gamma(
            gamma, self.gamma_min, self.gamma_max
        ).view(-1, 1)
        prediction = self.network(
            in_tensor,
            normalized_gamma=normalized_gamma,
            labels=batch["labels"],
            mesh=batch["mesh"]
        )

        ## Estimate loss
        error_diffusion = self.param.estimate_errors(
            prediction,
            in_data=noised_residual,
            target=residual,
            noise=noise,
            alpha=alpha,
            sigma=sigma,
            gamma=gamma,
        )
        weighted_error = self.weighting(gamma) * error_diffusion
        density = self.scheduler.get_density(gamma)
        loss = 0.5 * torch.mean(weighted_error / density)
        self.log(
            f'{prefix}/loss', loss,
            batch_size=in_tensor.size(0),
            prog_bar=True, sync_dist=sync_dist,
        )

        denoised = self.param(
            prediction=prediction,
            in_data=noised_residual,
            alpha=alpha,
            sigma=sigma,
            gamma=gamma,
        )
        mse_denoised = torch.mean((residual-denoised).pow(2))
        self.log_dict({
            f"{prefix}/data_loss": mse_denoised,
        }, batch_size=in_tensor.size(0), sync_dist=sync_dist)

        return {
            "loss": loss,
            "gamma": gamma,
            "weighted_diff_error": weighted_error
        }

    def on_train_batch_end(
            self,
            outputs: Dict[str, torch.Tensor],
            batch: Dict[str, torch.Tensor],
            batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        # Update scheduler
        scheduler_target = torch.mean(
            outputs["weighted_diff_error"].detach(),
            dim=(1, 2, 3)
        )
        self.scheduler.update(
            outputs["gamma"].view(-1),
            scheduler_target
        )
        return outputs

    def on_before_zero_grad(self, optimizer):
        self.ema_model.update_parameters(self.network)

    def on_train_end(self) -> None:
        torch.optim.swa_utils.update_bn(
            self.trainer.train_dataloader, self.ema_model
        )

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
