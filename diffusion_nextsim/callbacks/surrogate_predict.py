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
from typing import Dict, Any, Iterable

# External modules
import matplotlib
matplotlib.use('agg')

import lightning.pytorch as pl
import torch
import numpy as np
from numpy import ma
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.gridspec as mpl_gs
import matplotlib.colors as mpl_c
import cmocean

import cartopy
import cartopy.crs as ccrs

# Internal modules

main_logger = logging.getLogger(__name__)


class SurrogatePredictCallback(pl.Callback):
    def __init__(
            self,
            auxiliary_path: str,
            variables: Iterable[str],
            plot_idx: int = 0,
            n_input_steps: int = 2,
            every_n_step: int = 500,
            steps_to_plot: Iterable[int] = (0, 1, 4, 20, 60)
    ):
        super().__init__()
        self.ds_aux = xr.open_dataset(auxiliary_path)
        self.variables = variables
        self.steps_to_plot = steps_to_plot
        self.plot_idx = plot_idx
        self.n_input_steps = n_input_steps
        self.every_n_step = every_n_step
        self._batch = None

    def load_data(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule"
    ) -> Dict[str, torch.Tensor]:
        dataset = trainer.val_dataloaders.dataset
        idx_slice = slice(
            self.plot_idx,
            self.plot_idx+dataset.delta_t*(self.cycle_steps+self.n_input_steps),
            dataset.delta_t
        )
        state_traj = dataset.dataset["state_data"][idx_slice]
        state_traj = state_traj[:, dataset.state_idx]
        state_traj[..., dataset.mask_bool] = 0.
        state_traj = torch.as_tensor(
            state_traj[None], dtype=torch.float32, device=pl_module.device
        )
        if dataset.forcing_idx:
            forcing_traj = dataset.dataset["forcing_data"][idx_slice]
            forcing_traj = forcing_traj[:, dataset.forcing_idx]
            forcing_traj[..., dataset.mask_bool] = 0.
            forcing_traj = torch.as_tensor(
                forcing_traj[None], dtype=torch.float32, device=pl_module.device
            )
        else:
            forcing_traj = torch.zeros(
                (state_traj.shape[0],
                 self.cycle_steps+self.n_input_steps,
                 *state_traj.shape[-2:]),
                device=pl_module.device, dtype=torch.float32
            )
        mesh = torch.as_tensor(
            dataset.mesh[None], dtype=torch.float32, device=pl_module.device
        )
        mask = torch.as_tensor(
            dataset.mask[None], dtype=torch.float32, device=pl_module.device
        )
        return {
            "state_traj": state_traj,
            "forcing_traj": forcing_traj,
            "mesh": mesh,
            "mask": mask
        }

    def on_validation_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule"
    ) -> None:
        if self._batch is None and trainer.is_global_zero:
            self._batch = self.load_data(trainer, pl_module)

    @property
    def n_steps(self) -> int:
        return len(self.steps_to_plot)

    @property
    def cycle_steps(self) -> int:
        return max(self.steps_to_plot)

    def predict(
            self,
            pl_module: "pl.LightningModule"
    ) -> torch.Tensor:
        initial_conditions = self._batch["state_traj"][:, :self.n_input_steps]
        predictions = [
            split.squeeze(dim=1) for split in initial_conditions.split(1, dim=1)
        ]
        for k in range(self.n_input_steps, self.cycle_steps+self.n_input_steps):
            states = torch.stack(
                predictions[-self.n_input_steps:], dim=1
            )
            forcings = self._batch["forcing_traj"][:, k-self.n_input_steps:k+1]
            with torch.no_grad():
                curr_pred = pl_module(
                    states, forcings, mesh=self._batch["mesh"],
                    mask=self._batch["mask"]
                )
            predictions.append(curr_pred)
        predictions = torch.cat(predictions, dim=0)[self.n_input_steps-1:]
        return predictions

    def plot_fields(
            self,
            predictions: np.ndarray,
            trainer: "pl.Trainer",
            prefix: str = "val"
    ) -> None:
        figures = []
        for k, var_name in enumerate(self.variables):
            curr_pred = predictions[:, k]
            gs = mpl_gs.GridSpec(ncols=self.n_steps*10+1, nrows=1, top=0.8)
            fig = plt.figure(figsize=(self.n_steps*2, 2), dpi=150)
            abs_max = np.abs(curr_pred).max()
            for i, step in enumerate(self.steps_to_plot):
                ax = fig.add_subplot(
                    gs[i*10:(i+1)*10], projection=ccrs.NorthPolarStereo()
                )
                ax.set_facecolor(cmocean.cm.ice(0.))
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.spines.left.set_visible(False)
                ax.spines.right.set_visible(False)
                ax.spines.bottom.set_visible(False)
                if var_name == "sit":
                    cf = ax.pcolormesh(
                        self.ds_aux["longitude"],
                        self.ds_aux["latitude"],
                        ma.masked_array(
                            predictions[step, k], 1-self.ds_aux["mask"]
                        ),
                        cmap="cmo.ice", vmin=0, vmax=3,
                        shading="nearest", transform=ccrs.PlateCarree()
                    )
                elif var_name == "sic":
                    cf = ax.pcolormesh(
                        self.ds_aux["longitude"],
                        self.ds_aux["latitude"],
                        ma.masked_array(
                            predictions[step, k], 1 - self.ds_aux["mask"]
                        ),
                        cmap="cmo.ice", vmin=0, vmax=1,
                        shading="nearest", transform=ccrs.PlateCarree()
                    )
                elif var_name == "damage":
                    cf = ax.pcolormesh(
                        self.ds_aux["longitude"],
                        self.ds_aux["latitude"],
                        ma.masked_array(
                            1-predictions[step, k], 1-self.ds_aux["mask"]
                        ),
                        cmap="cmo.ice_r", vmin=0, vmax=1,
                        shading="nearest", transform=ccrs.PlateCarree()
                    )
                else:
                    cf = ax.pcolormesh(
                        self.ds_aux["longitude"],
                        self.ds_aux["latitude"],
                        ma.masked_array(
                            predictions[step, k], 1-self.ds_aux["mask"]
                        ),
                        cmap="RdBu_r",
                        shading="nearest", transform=ccrs.PlateCarree(),
                        norm=mpl_c.CenteredNorm(halfrange=abs_max)
                    )
                ax.add_feature(cartopy.feature.LAND, fc="xkcd:putty", zorder=98)
                ax.set_title(f"Iteration: {step:d}")
            cax = fig.add_subplot(gs[-1:])
            cbar = fig.colorbar(cf, cax=cax, orientation="vertical")
            fig.suptitle(f"{var_name:s}")
            figures.append(fig)
        trainer.logger.log_image(f"{prefix:s}/prediction", figures)
        plt.close("all")

    def on_validation_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        if self.every_n_step >= 1 and \
                trainer.global_step > 0 and \
                trainer.global_step % self.every_n_step == 0 and \
                trainer.is_global_zero:
            predictions = self.predict(pl_module)
            self.plot_fields(
                predictions.cpu().numpy(), trainer=trainer, prefix="val"
            )
