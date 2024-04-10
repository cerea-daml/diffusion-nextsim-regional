#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 30/11/2023
# Created for diffusion_nextsim
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
import math

# External modules
import torch
import torch.nn.functional as F

# Internal modules


main_logger = logging.getLogger(__name__)


_const_gauss_pdf = 1 / math.sqrt(2 * math.pi)
_const_gauss_crps = 1 / math.sqrt(math.pi)


__all__ = [
    "get_fft_stats",
    "estimate_spectrum",
    "estimate_rankhist",
    "estimate_crps_gauss",
    "estimate_crps_ens"
]


def get_fft_stats(target):
    n_points = target.size(-1)
    kfreq = torch.fft.fftfreq(n_points) * n_points
    kfreq2D = torch.meshgrid(kfreq, kfreq, indexing='ij')
    knrm = torch.sqrt(kfreq2D[0].pow(2) + kfreq2D[1].pow(2))
    knrm = knrm.expand(target.shape).to(target)
    kbins = torch.arange(0.5, n_points // 2 + 1, 1.0).to(target)
    return knrm, kbins


def estimate_spectrum(
        predictions: torch.Tensor,
        knrm: torch.Tensor,
        kbins: torch.Tensor
) -> torch.Tensor:
    fft_sq = torch.fft.fft2(predictions).abs().pow(2)
    bin_idx = torch.searchsorted(kbins, knrm.contiguous(), right=True)
    target_tensor = torch.zeros(
        *knrm.shape[:-2], len(kbins) + 1
    ).to(predictions)
    spectrum = target_tensor.scatter_add(
        -1,
        bin_idx.view(*knrm.shape[:-2], -1),
        fft_sq.view(*knrm.shape[:-2], -1)
    )[..., 1:-1]
    nbins = target_tensor.scatter_add(
        -1,
        bin_idx.view(*knrm.shape[:-2], -1),
        torch.ones_like(fft_sq).view(*knrm.shape[:-2], -1)
    )[..., 1:-1]
    spectrum *= torch.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2) / nbins
    return spectrum.mean(dim=0).sum(dim=0)


def estimate_rankhist(
        ens: torch.Tensor,
        target: torch.Tensor,
) -> torch.Tensor:
    ens_target = torch.cat((target[None, ...], ens), dim=0)
    argsorted_ens = ens_target.argsort(dim=0)
    ranks = (argsorted_ens == 0).float().argmax(dim=0)
    one_hots = F.one_hot(ranks, num_classes=ens_target.size(0)).float()
    mean_hist = torch.mean(one_hots, dim=(0, 3, 4))
    return mean_hist


def estimate_crps_gauss(
        ens_mean, ens_std, target
) -> torch.Tensor:
    ens_std = ens_std.clamp(min=1E-9)
    normed_diff = (target-ens_mean)/ens_std
    pdf = _const_gauss_pdf * torch.exp(-normed_diff.pow(2)*0.5)
    cdf = torch.special.ndtr(normed_diff)
    crps = ens_std * (
            normed_diff * (2 * cdf - 1) + 2 * pdf
            - _const_gauss_crps
    )
    return torch.mean(crps, dim=(3, 4)).sum(dim=0)


def estimate_crps_ens(
        ens: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    # Based on Pyro implementation
    # https://github.com/pyro-ppl/pyro/blob/dev/pyro/ops/stats.py and
    # Hersbach, 2000, Eq. 20
    N = ens.size(0)
    ens = ens.sort(dim=0).values
    diff = ens[1:] - ens[:-1]
    weight = torch.arange(1, N, device=ens.device) * torch.arange(
        N - 1, 0, -1, device=ens.device
    )
    weight = weight.reshape(weight.shape + (1,) * (diff.dim() - 1))
    crps = (
        (ens-target[None]).abs().mean(dim=0)
        -(weight * diff).sum(dim=0)/N**2
    )
    return torch.mean(crps, dim=(3, 4)).sum(dim=0)
