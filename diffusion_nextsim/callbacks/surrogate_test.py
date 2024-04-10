#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 07/11/2023
# Created for diffusion_nextsim
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Dict, Iterable, Any
import os.path

# External modules
import lightning.pytorch as pl
import torch
import pandas as pd

# Internal modules
from diffusion_nextsim.utils import *


main_logger = logging.getLogger(__name__)


class SurrogateTestCallback(pl.Callback):
    def __init__(
            self,
            variables: Iterable[str],
            n_input_steps: int = 1,
            n_cycles: int = 30,
            n_ens: int = 1,
            compile: bool = False,
            use_ema: bool = True
    ):
        super().__init__()
        self.variables = variables
        self.n_input_steps = n_input_steps
        self.n_cycles = n_cycles
        self.n_ens = n_ens
        self.compile = compile
        self.use_ema = use_ema
        self.test_scores = {}

    def log_rankhist(self, pl_module: "pl.LightningModule"):
        t, c, e = self.test_scores["rankhist"].shape
        rank_hist = self.test_scores["rankhist"].view(-1, e).T
        index = torch.arange(
            e, device=self.test_scores["rankhist"].device
        )[:, None]
        rank_hist = torch.cat((index, rank_hist), dim=1)
        rank_cols = ["rank"] + [
            f"{col[0]:d}it_{col[1]}"
            for col in pd.MultiIndex.from_product(
                [range(t), self.variables]
            )
        ]
        rank_df = pd.DataFrame(rank_hist, columns=rank_cols,)
        pl_module.logger.log_table(
            key="test/rank_hist", dataframe=rank_df
        )

    def log_spectrum(self, pl_module: "pl.LightningModule"):
        t, c, k = self.test_scores["spectrum"].shape
        spectrum_cols = ["k"] + [
            f"{col[0]:d}it_{col[1]}"
            for col in pd.MultiIndex.from_product(
                [range(t), self.variables]
            )
        ]
        spectrum_data = torch.cat((
            torch.arange(1, k+1)[:, None],
            self.test_scores["spectrum"].view(-1, k).T
        ), dim=1)
        spectrum_df = pd.DataFrame(
            spectrum_data, columns=spectrum_cols
        )
        pl_module.logger.log_table(
            key="test/spectrum", dataframe=spectrum_df
        )

        # Error spectrum
        t, c, k = self.test_scores["err_spectrum"].shape
        spectrum_cols = ["k"] + [
            f"{col[0]:d}it_{col[1]}"
            for col in pd.MultiIndex.from_product(
                [range(t), self.variables]
            )
        ]
        spectrum_data = torch.cat((
            torch.arange(1, k+1)[:, None],
            self.test_scores["err_spectrum"].view(-1, k).T
        ), dim=1)
        spectrum_df = pd.DataFrame(
            spectrum_data, columns=spectrum_cols
        )
        pl_module.logger.log_table(
            key="test/err_spectrum", dataframe=spectrum_df
        )

    def log_aggregated(self, pl_module: "pl.LightningModule"):
        data = [
            self.test_scores["mse"].sqrt(),
            self.test_scores["mae"],
            self.test_scores["bias"],
            self.test_scores["norm_dx"].sqrt(),
            self.test_scores["field_var"].sqrt(),
        ]
        score_names = ["rmse", "mae", "bias", "norm_dx", "field_std"]
        if self.n_ens > 1:
            data += [
                self.test_scores["ens_var"].sqrt(),
                self.test_scores["crps_gauss"],
                self.test_scores["crps_ens"],
            ]
            score_names += ["spread", "crps_gauss", "crps_ens"]
        combined_scores = torch.cat(
            [
                torch.arange(
                    self.test_scores["mse"].size(0),
                )[:, None]] + data,
            dim=1
        ).cpu()
        columns = [
            "_".join(col)
            for col in pd.MultiIndex.from_product(
                [score_names, self.variables]
            ).to_flat_index()
        ]
        columns = ["iterations"] + columns
        scores_frame = pd.DataFrame(
            combined_scores,
            columns=columns
        )
        pl_module.logger.log_table(
            key="test/scores", dataframe=scores_frame
        )

    def log_tensors(self, trainer: "pl.Trainer"):
        save_path = os.path.join(trainer.default_root_dir, "scores_tensors.pt")
        torch.save({
            "mse_field": self.test_scores["mse_field"],
            "mae_field": self.test_scores["mae_field"],
            "bias_field": self.test_scores["bias_field"]
        }, save_path)

    def evaluate_test(
            self,
            predictions: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        self.test_scores["n_samples"] = (self.test_scores["n_samples"]
                                         + predictions.size(1))

        # Ensemble mean scores
        pred_mean = predictions.mean(dim=0)
        error = pred_mean - target

        ## MSE
        mse = torch.mean(error.pow(2), dim=(3, 4)).sum(dim=0)
        self.test_scores["mse"] = self.test_scores["mse"] + mse

        ## MAE
        mae = torch.mean(error.abs(), dim=(3, 4)).sum(dim=0)
        self.test_scores["mae"] = self.test_scores["mae"] + mae

        ## Mean error
        bias = torch.mean(error, dim=(3, 4)).sum(dim=0)
        self.test_scores["bias"] = self.test_scores["bias"] + bias

        ## MSE over field
        mse_field = error.pow(2).sum(dim=0)
        self.test_scores["mse_field"] = (self.test_scores["mse_field"]
                                         + mse_field)

        ## MAE over field
        mae_field = error.abs().sum(dim=0)
        self.test_scores["mae_field"] = (self.test_scores["mae_field"]
                                         + mae_field)

        ## Mean error over field
        bias_field = error.sum(dim=0)
        self.test_scores["bias_field"] = (self.test_scores["bias_field"]
                                          + bias_field)

        # General field statistics
        ## Squared norm of the dynamics
        norm_dx = predictions[:, :, 1:]-predictions[:, :, :-1]
        norm_dx = torch.mean(norm_dx.pow(2), dim=(0, 4, 5)).sum(dim=0)
        norm_dx = torch.cat((
            torch.zeros(1, norm_dx.size(-1), device=norm_dx.device),
            norm_dx
        ), dim=0)
        self.test_scores["norm_dx"] = self.test_scores["norm_dx"] + norm_dx

        ## Variance of the fields
        field_mean = torch.mean(predictions, dim=(4, 5))
        field_sq_mean = torch.mean(predictions.pow(2), dim=(4, 5))
        field_var = field_sq_mean-field_mean.pow(2)
        field_var = field_var.mean(dim=0).sum(dim=0)
        self.test_scores["field_var"] = (self.test_scores["field_var"]
                                         + field_var)

        ## Spectrum
        knrm, kbins = get_fft_stats(predictions)
        spectrum = estimate_spectrum(predictions, knrm, kbins)
        self.test_scores["spectrum"] = self.test_scores["spectrum"] + spectrum

        ## Error spectrum
        knrm, kbins = get_fft_stats(error[None])
        err_spectrum = estimate_spectrum(error[None], knrm, kbins)
        self.test_scores["err_spectrum"] = (self.test_scores["err_spectrum"]
                                            + err_spectrum)

        # Ensemble scores
        if self.n_ens > 1:
            pred_var = predictions.var(dim=0)
            score_var = torch.mean(pred_var, dim=(3, 4)).sum(dim=0)
            self.test_scores["ens_var"] = (self.test_scores["ens_var"]
                                           + score_var)
            crps_gauss = estimate_crps_gauss(pred_mean, pred_var.sqrt(), target)
            self.test_scores["crps_gauss"] = (self.test_scores["crps_gauss"]
                                              + crps_gauss)
            crps_ens = estimate_crps_ens(predictions, target)
            self.test_scores["crps_ens"] = (self.test_scores["crps_ens"]
                                              + crps_ens)
            mean_rankhist = estimate_rankhist(predictions, target)
            self.test_scores["rankhist"] = (self.test_scores["rankhist"]
                                            + mean_rankhist)
        return mse[1, 0]

    def on_test_epoch_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule"
    ) -> None:
        self.test_scores = {
            "mse": 0,
            "mae": 0,
            "bias": 0,
            "norm_dx": 0,
            "field_var": 0,
            "ens_var": 0,
            "crps_gauss": 0,
            "crps_ens": 0,
            "mse_field": 0,
            "mae_field": 0,
            "bias_field": 0,
            "spectrum": 0,
            "err_spectrum": 0,
            "rankhist": 0,
            "n_samples": 0
        }

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        b, t, c, h, w = batch["state_traj"].shape
        predictions = [
            split.squeeze(dim=1).expand(self.n_ens, b, c, h, w)
            for split in batch["state_traj"][:, :self.n_input_steps].split(1, dim=1)
        ]
        for k in range(self.n_input_steps, batch["state_traj"].size(1)):
            states = torch.stack(
                predictions[-self.n_input_steps:], dim=2
            ).reshape(b * self.n_ens, self.n_input_steps, c, h, w)
            forcings = batch["forcing_traj"][:, k - self.n_input_steps:k + 1]
            forcings = forcings.expand(
                self.n_ens, b, self.n_input_steps + 1, -1, h, w
            ).reshape(b * self.n_ens, self.n_input_steps+1, -1, h, w)
            mesh = batch["mesh"].expand(
                self.n_ens, b, 3, h, w
            ).reshape(b * self.n_ens, 3, h, w)
            mask = batch["mask"].expand(
                self.n_ens, b, 1, h, w
            ).reshape(b * self.n_ens, 1, h, w)
            with torch.no_grad():
                curr_pred = pl_module(
                    states, forcings, mesh=mesh, mask=mask
                )
            curr_pred = curr_pred.reshape(self.n_ens, b, c, h, w)
            predictions.append(curr_pred)
        predictions = torch.stack(predictions, dim=2)[:, :, self.n_input_steps-1:]
        loss = self.evaluate_test(
            predictions.cpu(),
            batch["state_traj"][:, self.n_input_steps-1:].cpu(),
        )
        pl_module.log(
            "loss",
            self.test_scores["mae"][1, 0]/self.test_scores["n_samples"],
            prog_bar=True, on_step=True
        )
        return loss

    def on_test_epoch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule"
    ) -> None:
        # Average test scores
        self.test_scores = {
            name: score/self.test_scores["n_samples"]
            for name, score in self.test_scores.items()
            if name != "n_samples"
        }

        # Log the scores
        self.log_aggregated(pl_module)
        self.log_spectrum(pl_module)
        self.log_tensors(trainer)
        if self.n_ens > 1:
            self.log_rankhist(pl_module)
