#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 06/10/2023
# Created for diffusion_nextsim
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Iterable, Dict, Tuple

# External modules
import xarray as xr
import cartopy.crs as ccrs

import zarr

import numpy as np

# Internal modules

main_logger = logging.getLogger(__name__)


def get_mesh(ds_aux: xr.Dataset, length_scale: float = 350000) -> np.ndarray:
    plate_carree = ccrs.PlateCarree()
    stereo = ccrs.NorthPolarStereo()
    mesh = stereo.transform_points(
        plate_carree,
        ds_aux["longitude"].values,
        ds_aux["latitude"].values
    ).transpose(2, 0, 1) / length_scale
    return mesh.astype(np.float32)


def estimate_rot2curv(
        ds_aux: xr.Dataset
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates the rotation from vectors in lat/lon into curvilinear coordinates,
    as given by the sea-ice simulations. The function is inspired by
    the "geo2ocean" module from the NEMO model "/src/OCE/SBC/geo2ocean.F90".
    Returns the sin and cosine of the rotation for the direct application on
    U and V.
    """
    plate_carree = ccrs.PlateCarree()
    stereo = ccrs.NorthPolarStereo()
    mesh = stereo.transform_points(
        plate_carree,
        ds_aux["longitude"].values,
        ds_aux["latitude"].values
    ).transpose(2, 0, 1)[:2]

    x_north = -mesh[0]
    y_north = -mesh[1]
    dist_sq_north = x_north**2 + y_north**2

    x_diff = mesh[0, 1:, :]-mesh[0, :-1, :]
    y_diff = mesh[1, 1:, :]-mesh[1, :-1, :]
    # Pad in the south to get original shape
    x_diff = np.concatenate((x_diff[..., [0], :], x_diff), axis=-2)
    y_diff = np.concatenate((y_diff[..., [0], :], y_diff), axis=-2)

    normalizer = np.sqrt((x_diff**2 + y_diff**2) * dist_sq_north)
    normalizer = normalizer.clip(min=1E-7)

    sin_rot = (x_north*y_diff - y_north*x_diff) / normalizer
    cos_rot = (x_north*x_diff + y_north*y_diff) / normalizer

    # To increase stability where the difference between longitudes is small
    lon_diff = (
        ds_aux["longitude"].values[..., 1:, :]
        -ds_aux["longitude"].values[..., :-1, :]
    ) % 360
    lon_diff = np.concatenate((lon_diff[..., [0], :], lon_diff), axis=-2)
    lon_mask = lon_diff > 1E-8
    sin_rot = np.where(lon_mask, sin_rot, 0.)
    cos_rot = np.where(lon_mask, cos_rot, 1.)
    return sin_rot, cos_rot


def rotate_uv2curv(
        u_array: np.ndarray,
        v_array: np.ndarray,
        sin_rot: np.ndarray,
        cos_rot: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the rotation to u and v with given sine and cosine of the rotation
    angle.
    Returns the rotated U and V.
    """
    u_rot = u_array * cos_rot + v_array * sin_rot
    v_rot = v_array * cos_rot - u_array * sin_rot
    return u_rot, v_rot


def load_zarr(
        zarr_path: str,
        variables: Iterable[str, ]
) -> Dict[str, np.ndarray]:
    dataset = zarr.open(zarr_path, mode="r")
    return {var: dataset[var][...] for var in variables}
