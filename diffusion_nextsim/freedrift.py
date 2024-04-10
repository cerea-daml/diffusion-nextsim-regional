#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 13/02/2024
# Created for diffusion_nextsim
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2024}  {Tobias Sebastian Finn}

# System modules
import logging


# External modules
import torch.nn
from torch.nn.functional import pad

import numpy as np

# Internal modules


main_logger = logging.getLogger(__name__)


__all__ = ["OnlyAtmosphereModule", "SeaIceVelocityModule", "FreedriftModel"]


class SeaIceVelocityModule(torch.nn.Module):
    def forward(
            self,
            atm_forcing: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        return atm_forcing[..., :2, :, :]


class OnlyAtmosphereModule(torch.nn.Module):
    def __init__(
            self,
            alpha: float = 0.0174,
            theta: float = -25.,
    ):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(alpha))
        self.theta = torch.nn.Parameter(torch.tensor(np.deg2rad(theta)))

    def forward(
            self,
            atm_forcing: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        atm_speed = (
            atm_forcing[..., 0, :, :]**2 + atm_forcing[..., 1, :, :]**2
        ).sqrt()
        atm_angle = torch.arctan2(
            atm_forcing[..., 1, :, :],
            atm_forcing[..., 0, :, :]
        )
        out_tensor = torch.stack([
            self.alpha * atm_speed * torch.cos(atm_angle + self.theta),
            self.alpha * atm_speed * torch.sin(atm_angle + self.theta),
        ], dim=-3)
        return out_tensor


def _interp_time(
        curr_time: torch.Tensor,
        forcings: torch.Tensor
) -> torch.Tensor:
    floor_time = torch.floor(curr_time).int()
    ceil_time = torch.ceil(curr_time).int()
    weights = torch.stack([(ceil_time - curr_time), (curr_time - floor_time)])
    # To circumvent zero division
    weights = weights + 1E-9
    weights = weights / weights.sum(dim=-1)
    return (
            forcings[..., [int(floor_time), int(ceil_time)], :, :, :]
            * weights[:, None, None, None]
    ).sum(dim=-4)


def _estimate_distances(mesh):
    dx = torch.sqrt((torch.diff(mesh, dim=-1) ** 2).sum(dim=-3))
    dx = torch.cat(
        (
            dx[..., [0]],
            dx,
            dx[..., [-1]]
        ), dim=-1
    )
    dx = torch.cat((dx[..., [0], :], dx), dim=-2)
    dy = torch.sqrt((torch.diff(mesh, dim=-2) ** 2).sum(dim=-3))
    dy = torch.cat(
        (
            dy[..., [0], :],
            dy,
            dy[..., [-1], :]
        ), dim=-2
    )
    dy = torch.cat((dy[..., [0]], dy), dim=-1)
    return torch.stack((dx, dy), dim=-3)


def _interp_nearest(
        target_grid: torch.Tensor,
        source_values: torch.Tensor,
):
    idx_bounds = target_grid.clamp(min=0, max=63).round().int()
    idx_bounds = idx_bounds.reshape(2, -1)
    interp_values = source_values[:, idx_bounds[1], idx_bounds[0]]
    return interp_values.reshape(source_values.size(0), 64, 64)


def _interp_linear(
        target_grid: torch.Tensor,
        source_values: torch.Tensor,
) -> torch.Tensor:
    # Bilinear algorithm
    # From https://en.wikipedia.org/wiki/Bilinear_interpolation
    target_grid = target_grid.clamp(min=0, max=63)
    grid_floor = target_grid.floor().int()
    grid_norm = target_grid-grid_floor
    grid_ceil = target_grid.ceil().int()
    a0 = source_values[:, grid_floor[1], grid_floor[0]]
    a10 = (
        source_values[:, grid_floor[1], grid_ceil[0]]
        -source_values[:, grid_floor[1], grid_floor[0]]
    )
    a01 = (
        source_values[:, grid_ceil[1], grid_floor[0]]
        -source_values[:, grid_floor[1], grid_floor[0]]
    )
    a11 = (
        source_values[:, grid_ceil[1], grid_ceil[0]]
        -source_values[:, grid_floor[1], grid_ceil[0]]
        -source_values[:, grid_ceil[1], grid_floor[0]]
        +source_values[:, grid_floor[1], grid_floor[0]]
    )
    return (
        a0
        +a10*grid_norm[0]
        +a01*grid_norm[1]
        +a11*grid_norm[0]*grid_norm[1]
    )


def _interp_cubic(
        t: torch.Tensor,
        fm1: torch.Tensor,
        f0: torch.Tensor,
        f1: torch.Tensor,
        f2: torch.Tensor
) -> torch.Tensor:
    # Bicubic convolution algorithm
    # From https://en.wikipedia.org/wiki/Bicubic_interpolation
    return 0.5 * (
        2 * f0
        + (f1-fm1) * t
        + (2*fm1-5*f0+4*f1-f2) * t**2
        + (-fm1+3*f0-3*f1+f2) * t**3
    )


def _interp_bicubic(
        target_grid: torch.Tensor,
        source_values: torch.Tensor,
) -> torch.Tensor:
    # Bicubic convolution algorithm
    # From https://en.wikipedia.org/wiki/Bicubic_interpolation
    clipped = target_grid.clamp(min=0, max=63)
    origin = clipped.floor().int()
    normed = clipped - origin

    # To allow values outside of target grid
    src_pad = pad(source_values, pad=(1, 2, 1, 2), mode="replicate")
    bm1 = _interp_cubic(
        normed[0],
        src_pad[:, origin[1], origin[0]],
        src_pad[:, origin[1], origin[0]+1],
        src_pad[:, origin[1], origin[0]+2],
        src_pad[:, origin[1], origin[0]+3],
    )
    b0 = _interp_cubic(
        normed[0],
        src_pad[:, origin[1]+1, origin[0]],
        src_pad[:, origin[1]+1, origin[0]+1],
        src_pad[:, origin[1]+1, origin[0]+2],
        src_pad[:, origin[1]+1, origin[0]+3],
    )
    b1 = _interp_cubic(
        normed[0],
        src_pad[:, origin[1]+2, origin[0]],
        src_pad[:, origin[1]+2, origin[0]+1],
        src_pad[:, origin[1]+2, origin[0]+2],
        src_pad[:, origin[1]+2, origin[0]+3],
    )
    b2 = _interp_cubic(
        normed[0],
        src_pad[:, origin[1]+3, origin[0]],
        src_pad[:, origin[1]+3, origin[0]+1],
        src_pad[:, origin[1]+3, origin[0]+2],
        src_pad[:, origin[1]+3, origin[0]+3],
    )
    return _interp_cubic(normed[1], bm1, b0, b1, b2)


_avail_interp = {
    "nearest": _interp_nearest,
    "linear": _interp_linear,
    "cubic": _interp_bicubic,
}


class FreedriftModel(torch.nn.Module):
    def __init__(
            self,
            velocity_module: torch.nn.Module,
            dt_model: float = 600.,
            dt_forcing: float = 21600.,
            interp_mode: str = "linear",
    ):
        super().__init__()
        self.velocity_module = velocity_module
        self.dt_model = dt_model
        self.steps_between = int(dt_forcing/dt_model)
        self.interp_func = _avail_interp[interp_mode]

    def forward(
            self,
            initial_conditions: torch.Tensor,
            forcings: torch.Tensor,
            mesh: torch.Tensor
    ) -> torch.Tensor:
        # Estimate distances
        distances = _estimate_distances(mesh)

        # Initialize position
        curr_pos = torch.stack(
            torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij"),
        ).swapdims(1, 2)
        vel_pos = curr_pos = curr_pos.expand(
            *initial_conditions.shape[:-3], 2, 64, 64
        )

        # Start iteration
        steps = torch.arange(0, (forcings.size(-4) - 1) * self.steps_between)
        steps = torch.flip(steps, dims=(-1, ))
        for step in steps:
            # Estimate sea-ice velocities
            atm_forcing = _interp_time(
                (step + 0.5) / self.steps_between,
                forcings[..., :2, :, :]
            )
            velocities = self.velocity_module(
                atm_forcing, forcings[..., 2:, :, :]
            )

            # Interpolate velocity
            ##  Nearest neighbor interpolation (efficient by rounding)
            vel_idx = vel_pos.clamp(
                min=torch.tensor([0, 0])[:, None, None],
                max=torch.tensor([63, 63])[:, None, None]
            ).round().int()
            curr_vel = torch.stack((
                velocities[..., 0, :, :][..., vel_idx[1], vel_idx[0]],
                velocities[..., 1, :, :][..., vel_idx[1], vel_idx[0]],
            ), dim=-3)

            ##  Convert from stereographic meters to curvilinear unit
            curr_vel = curr_vel / torch.stack((
                distances[..., 0, :, :][..., vel_idx[1], vel_idx[0]],
                distances[..., 1, :, :][..., vel_idx[1], vel_idx[0]],
            ), dim=-3)

            # Euler step to estimate displacement from velocity
            displacement = -self.dt_model * curr_vel

            # Integrate position
            curr_pos = curr_pos + displacement
            vel_pos = curr_pos + 1.5 * displacement

        interp_state = self.interp_func(curr_pos, initial_conditions)
        return interp_state
