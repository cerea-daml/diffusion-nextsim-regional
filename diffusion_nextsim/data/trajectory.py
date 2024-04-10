#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 08/09/2023
# Created for diffusion_nextsim
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Iterable, Dict

# External modules
import zarr
import xarray as xr
import numpy as np

# Internal modules
from .augmentation import Augmentation
from .zip_dataset import ZipDataset
from .utils import get_mesh, estimate_rot2curv, rotate_uv2curv

main_logger = logging.getLogger(__name__)


class TrajectoryDataset(ZipDataset):
    def __init__(
            self,
            zarr_path: str,
            aux_path: str,
            augmentation: Augmentation = Augmentation(),
            delta_t: int = 2,
            n_cycles: int = 30,
            state_variables: Iterable[str] = (
                    "sit", "sic", "damage", "siu", "siv"
            ),
            forcing_variables: Iterable[str] = ("tus", "huss", "uas", "vas"),
            rotate_wind: bool = True,
            lengthscale: float = 1.,
            start_date: str = None,
            zip_path: str = None,
            extract: bool = True,
            fast: bool = True,
    ):
        super().__init__(
            data_path=zarr_path, zip_path=zip_path, extract=extract, fast=fast
        )
        self.dataset = zarr.open(zarr_path)
        state_var_names = list(self.dataset["var_names_1"][:])
        self.state_idx = [state_var_names.index(var) for var in state_variables]
        forcing_var_names = list(self.dataset["var_names_2"][:])
        self.forcing_idx = [
            forcing_var_names.index(var) for var in forcing_variables
        ]
        self.start_idx = 0
        if start_date is not None:
            with xr.open_zarr(zarr_path) as ds_time:
                time_index = ds_time.indexes["time"]
                self.start_idx = time_index.get_loc(start_date)
        self.rotate_wind = rotate_wind
        try:
            self.wind_idx = [
                self.forcing_idx.index("uas"), self.forcing_idx.index("vas")
            ]
        except ValueError:
            self.wind_idx = None
        self.delta_t = delta_t
        self.n_cycles = n_cycles
        self.augmentation = augmentation
        with xr.open_dataset(aux_path) as ds_aux:
            self.mesh = get_mesh(ds_aux, length_scale=lengthscale)
            self.mask_bool = ~ds_aux["mask"].astype(bool)
            self.mask = ds_aux["mask"].values[None]
            self.sin_rot, self.cos_rot = estimate_rot2curv(ds_aux)

    def __len__(self) -> int:
        return (self.dataset["time"].shape[0]
                -self.start_idx
                -self.delta_t*self.n_cycles)

    def __getitem__(self, idx) -> Dict[str, "torch.Tensor"]:
        idx_slice = slice(
            self.start_idx + idx,
            self.start_idx + idx + self.delta_t * self.n_cycles,
            self.delta_t
        )
        state_traj = self.dataset["state_data"][idx_slice]
        state_traj = state_traj[:, self.state_idx]
        state_traj[..., self.mask_bool] = 0.

        if self.forcing_idx:
            forcing_traj = self.dataset["forcing_data"][idx_slice]
            forcing_traj = forcing_traj[:, self.forcing_idx]
            forcing_traj[..., self.mask_bool] = 0.
            if self.rotate_wind and self.wind_idx is not None:
                forcing_traj[:, self.wind_idx] = np.stack(rotate_uv2curv(
                    forcing_traj[:, self.wind_idx[0]],
                    forcing_traj[:, self.wind_idx[1]],
                    self.sin_rot,
                    self.cos_rot
                ), axis=1)
        else:
            forcing_traj = np.zeros(
                ((self.n_cycles+1), 1, *state_traj.shape[-2:]),
                dtype=np.float32
            )

        return self.augmentation(
            state_data=state_traj,
            forcing_data=forcing_traj,
            mask=self.mask,
            mesh=self.mesh
        )
