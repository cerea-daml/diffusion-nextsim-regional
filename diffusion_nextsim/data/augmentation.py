#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 10/10/2023
# Created for diffusion_nextsim
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Tuple, Dict

# External modules
import torch
import torch.nn
from torch.nn.functional import pad
import torchvision.transforms.functional as f

import numpy as np

# Internal modules

main_logger = logging.getLogger(__name__)


class Augmentation(object):
    def __init__(
            self,
            hflip_prob: float = 0.,
            vflip_prob: float = 0.,
            rot_prob: float = 0.,
            crop_fields: bool = False,
            crop_sizes: Tuple[int, int] = (64, 64),
            dtype: torch.dtype = torch.float32,
    ):
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rot_prob = rot_prob
        self.crop_fields = crop_fields
        self.crop_sizes = crop_sizes
        self.dtype = dtype

    def pad_array(self, array: torch.Tensor) -> torch.Tensor:
        return pad(
            array,
            (
                int(self.crop_sizes[0]//2), int(self.crop_sizes[0]//2),
                int(self.crop_sizes[1]//2), int(self.crop_sizes[1]//2)
            )
        )

    def __call__(
            self,
            state_data: np.ndarray,
            forcing_data: np.ndarray,
            mask: np.ndarray,
            mesh: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        # List to hold labels for conditioning
        labels = []

        # To tensor
        state_data = torch.as_tensor(state_data, dtype=self.dtype)
        forcing_data = torch.as_tensor(forcing_data, dtype=self.dtype)
        mask = torch.as_tensor(mask, dtype=self.dtype)
        mesh = torch.as_tensor(mesh, dtype=self.dtype)

        # Horizontal flip
        h_flip = torch.rand(1) < self.hflip_prob
        if h_flip:
            state_data = f.hflip(state_data)
            forcing_data = f.hflip(forcing_data)
            mask = f.hflip(mask)
            mesh = f.hflip(mesh)
        labels.append(h_flip)

        # Vertical flip
        v_flip = torch.rand(1) < self.vflip_prob
        if v_flip:
            state_data = f.vflip(state_data)
            forcing_data = f.vflip(forcing_data)
            mask = f.vflip(mask)
            mesh = f.vflip(mesh)
        labels.append(v_flip)

        # Rotation
        rot = torch.rand(1) < self.rot_prob
        if rot:
            state_data = f.rotate(state_data, angle=90)
            forcing_data = f.rotate(forcing_data, angle=90)
            mask = f.rotate(mask, angle=90)
            mesh = f.rotate(mesh, angle=90)
        labels.append(rot)

        # Crop
        if self.crop_fields:
            top_left = (
                torch.randint(state_data.shape[-2], (1,)).item(),
                torch.randint(state_data.shape[-1], (1,)).item()
            )
            slices = (
                slice(top_left[0], top_left[0]+self.crop_sizes[0]),
                slice(top_left[1], top_left[1]+self.crop_sizes[1])
            )
            state_data = self.pad_array(state_data)[..., slices[0], slices[1]]
            forcing_data = self.pad_array(forcing_data)[..., slices[0], slices[1]]
            mask = self.pad_array(mask)[..., slices[0], slices[1]]
            mesh = self.pad_array(mesh)[..., slices[0], slices[1]]

        # Convert labels to tensor
        labels = torch.as_tensor(labels, dtype=self.dtype)
        return {
            "state_traj": state_data,
            "forcing_traj": forcing_data,
            "mask": mask,
            "mesh": mesh,
            "labels": labels
        }
