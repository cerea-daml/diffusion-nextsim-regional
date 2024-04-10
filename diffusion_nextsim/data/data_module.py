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
from typing import Iterable
import os.path

# External modules
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split

# Internal modules
from .augmentation import Augmentation
from .trajectory import TrajectoryDataset


main_logger = logging.getLogger(__name__)


class SurrogateDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_path: str,
            aux_path: str,
            augmentation: Augmentation,
            state_variables: Iterable[str] = ("sit", ),
            forcing_variables: Iterable[str] = ("t2m", "d2m", "u10m", "v10m"),
            rotate_wind: bool = True,
            delta_t: int = 2,
            n_input_steps: int = 1,
            n_rollout_steps: int = 1,
            batch_size: int = 64,
            n_train_samples: int = None,
            train_start_date: str = None,
            lengthscale: float = 350000.,
            n_workers: int = 4,
            pin_memory: bool = True,
            zip_path: str = None,
            fast: bool = True,
            suffix: str = "",
    ):
        super().__init__()
        self._train_dataset = None
        self._val_dataset = None
        self._predict_dataset = None
        self._test_dataset = None
        self.rotate_wind = rotate_wind
        self.delta_t = delta_t
        self.n_input_steps = n_input_steps
        self.n_rollout_steps = n_rollout_steps
        self.data_path = data_path
        self.aux_path = aux_path
        self.forcing_variables = forcing_variables
        self.state_variables = state_variables
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.train_start_date = train_start_date
        self.n_train_samples = n_train_samples
        self.lengthscale = lengthscale
        self.n_workers = n_workers
        self.pin_memory = pin_memory
        self.zip_path = zip_path
        self.fast = fast
        self.suffix = suffix

    def setup(self, stage: str) -> None:
        if stage == "fit":
            file_name = f"train{self.suffix}.zarr"
            self._train_dataset = TrajectoryDataset(
                os.path.join(self.data_path, file_name),
                aux_path=self.aux_path,
                delta_t=self.delta_t,
                n_cycles=self.n_input_steps+self.n_rollout_steps,
                state_variables=self.state_variables,
                forcing_variables=self.forcing_variables,
                rotate_wind=self.rotate_wind,
                start_date=self.train_start_date,
                augmentation=self.augmentation,
                lengthscale=self.lengthscale,
                zip_path=os.path.join(
                    self.zip_path, f"{file_name:s}.zip"
                ) if self.zip_path is not None else None,
                fast=self.fast,
                extract=True
            )
            if self.n_train_samples is not None:
                self._train_dataset, _ = random_split(
                    self._train_dataset, (
                        self.n_train_samples,
                        len(self._train_dataset)-self.n_train_samples
                    )
                )
        if stage in ("fit", "validate"):
            file_name = f"validation{self.suffix}.zarr"
            self._val_dataset = TrajectoryDataset(
                os.path.join(self.data_path, file_name),
                aux_path=self.aux_path,
                delta_t=self.delta_t,
                n_cycles=self.n_input_steps+self.n_rollout_steps,
                state_variables=self.state_variables,
                forcing_variables=self.forcing_variables,
                rotate_wind=self.rotate_wind,
                augmentation=Augmentation(),
                lengthscale=self.lengthscale,
                zip_path=os.path.join(
                    self.zip_path, f"{file_name:s}.zip"
                ) if self.zip_path is not None else None,
                fast=self.fast,
                extract=True
            )
        elif stage in ("test", "predict"):
            file_name = f"test{self.suffix}.zarr"
            self._test_dataset = TrajectoryDataset(
                os.path.join(self.data_path, file_name),
                aux_path=self.aux_path,
                delta_t=self.delta_t,
                n_cycles=self.n_input_steps+1,
                state_variables=self.state_variables,
                forcing_variables=self.forcing_variables,
                rotate_wind=self.rotate_wind,
                augmentation=Augmentation(),
                lengthscale=self.lengthscale,
                zip_path=os.path.join(
                    self.zip_path, f"{file_name:s}.zip"
                ) if self.zip_path is not None else None,
                fast=self.fast,
                extract=True
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset, batch_size=self.batch_size, shuffle=True,
            pin_memory=self.pin_memory, num_workers=self.n_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.n_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.n_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.n_workers,
        )
