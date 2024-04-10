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

# External modules
import xarray as xr
import numpy as np

# Internal modules


logger = logging.getLogger(__name__)


def estimate_grads(ds_states, dx=12500):
    u_grad = xr.concat(
        (ds_states["siu"].diff("x"), ds_states["siu"].diff("y")),
        dim="components"
    ) / dx
    v_grad = xr.concat(
        (ds_states["siv"].diff("x"), ds_states["siv"].diff("y")),
        dim="components"
    ) / dx
    ds_grad = xr.Dataset({"u_grad": u_grad, "v_grad": v_grad})
    ds_grad = ds_grad.assign_coords(components=["x", "y"])
    return ds_grad


def estimate_deform(ds_states, dx=12500):
    ds_grad = estimate_grads(ds_states, dx=dx)
    deform_div = ds_grad["u_grad"].sel(components="x") + ds_grad["v_grad"].sel(components="y")
    deform_shear = np.sqrt(
        (
                ds_grad["u_grad"].sel(components="x") - ds_grad["v_grad"].sel(components="y")
        )**2 + (
                ds_grad["u_grad"].sel(components="y") + ds_grad["v_grad"].sel(components="x")
        )**2
    )
    deform_total = np.sqrt(deform_div ** 2 + deform_shear ** 2)
    ds_deform = xr.Dataset({
        "deform_div": deform_div,
        "deform_shear": deform_shear,
        "deform_tot": deform_total
    })
    return ds_deform
