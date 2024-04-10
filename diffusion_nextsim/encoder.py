#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 29/01/2024
# Created for diffusion_nextsim
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2024}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Iterable

# External modules
import torch

# Internal modules

main_logger = logging.getLogger(__name__)


class GaussianEncoder(torch.nn.Module):
    def __init__(
            self,
            mean: Iterable[float] = (0., ),
            std: Iterable[float] = (1., ),
            eps: float = 1E-9
    ):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean)[:, None, None])
        self.register_buffer("std", torch.tensor(std)[:, None, None])
        self.eps = eps

    def forward(
            self,
            in_tensor: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        normed_tensor = (in_tensor-self.mean) / (self.std + self.eps)
        return normed_tensor
