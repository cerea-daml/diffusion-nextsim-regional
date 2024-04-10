#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 15/02/2024
# Created for diffusion_nextsim
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2024}  {Tobias Sebastian Finn}

# System modules
import logging
import math
from types import MethodType
from typing import Iterable, Callable, Tuple

# External modules
import torch

# Internal modules
from ddm_dynamical.decoder.base_decoder import BaseDecoder
from ddm_dynamical.decoder.prediction_funcs import delta_prediction

main_logger = logging.getLogger(__name__)


class GaussianDecoder(BaseDecoder):
    def __init__(
            self,
            mean: Iterable[float] = (0., ),
            std: Iterable[float] = (1., ),
            lower_bound:  Iterable[float] = (-math.inf, ),
            upper_bound:  Iterable[float] = (math.inf, ),
            prediction_func: Callable = delta_prediction,
            **kwargs
    ):
        super().__init__(stochastic=False)
        self.register_buffer(
            "mean", torch.tensor(mean)[:, None, None]
        )
        self.register_buffer(
            "std", torch.tensor(std)[:, None, None]
        )
        self.register_buffer(
            "lower_bound", torch.tensor(lower_bound)[:, None, None]
        )
        self.register_buffer(
            "upper_bound", torch.tensor(upper_bound)[:, None, None]
        )
        self.to_prediction = MethodType(prediction_func, self)

    def forward(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        prediction = self.to_prediction(in_tensor, first_guess)
        return prediction.clamp(min=self.lower_bound, max=self.upper_bound)

    def loss(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            target: torch.Tensor,
            *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prediction = self.to_prediction(in_tensor, first_guess)
        error = (target-prediction) / self.std
        mse = torch.mean(error.pow(2))
        return mse, mse


class FFTSampler(torch.nn.Module):
    def __init__(
            self,
            periodic_spectrum: torch.Tensor = None,
            shift: torch.Tensor = None,
            cov: torch.Tensor = None,
            normalize: bool = True
    ):
        super().__init__()
        if periodic_spectrum is None:
            periodic_spectrum = torch.zeros(5, 64, 64)
            periodic_spectrum[:, 0, 0] = 1.
        self.register_buffer("periodic_spectrum", periodic_spectrum)
        if shift is None:
            shift = torch.zeros(5, 64, 64)
        self.register_buffer("shift", shift)
        if cov is None:
            cov = torch.eye(5)
        factor = torch.linalg.cholesky(cov)
        self.register_buffer("factor", factor)
        self.normalize = normalize

    def forward(self, prediction: torch.Tensor):
        noise = torch.randn_like(prediction)
        noise_fft = torch.fft.fft2(noise)
        noise_field = self.periodic_spectrum * noise_fft
        correlated = torch.fft.ifft2(noise_field).real
        correlated = correlated + self.shift
        if self.normalize:
            correlated = (
                correlated - correlated.mean(dim=0)
            ) / correlated.std(dim=0)
        return torch.einsum("ji,...ikl->...jkl", self.factor, correlated)


class StochasticDecoder(GaussianDecoder):
    def __init__(
            self,
            sampling_func: Callable,
            mean: Iterable[float] = (0., ),
            std: Iterable[float] = (1., ),
            lower_bound:  Iterable[float] = (-math.inf, ),
            upper_bound:  Iterable[float] = (math.inf, ),
            prediction_func: Callable = delta_prediction,
            **kwargs
    ):
        super().__init__(
            mean=mean, std=std,
            lower_bound=lower_bound, upper_bound=upper_bound,
            prediction_func=prediction_func, **kwargs
        )
        self.stochastic = True
        self.sampling_func = sampling_func

    def forward(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        prediction = self.to_prediction(in_tensor, first_guess)
        prediction.add_(self.sampling_func(prediction))
        return prediction.clamp(min=self.lower_bound, max=self.upper_bound)
