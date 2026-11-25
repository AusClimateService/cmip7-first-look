#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate seasonal means of tas and other vars over various historical periods

# full time series 1991 - 2020 will be available from historical run

Data has been transferred to Qld HPC from NCI. Maintained NCI file structure (from below institution).

Created on 03/11/2026

@author: schapman
"""
#%%#import packages
import numpy as np
import glob
import xarray as xr
import os
from pathlib import Path
import re
import xesmf as xe

#%% Functions

def regrid_to_target(da, target_grid):

    regridder = xe.Regridder(
        da,
        target_grid,
        "bilinear",
        periodic=False,
        reuse_weights=False
    )

    return regridder(da)

def find_models(base_dir):
    """
    Return dictionary:

    {
        model_name : {
            variant1 : variant_path,
            variant2 : variant_path
        }
    }
    """

    models = {}

    for institution_dir in base_dir.iterdir():

        if not institution_dir.is_dir():
            continue

        for model_dir in institution_dir.iterdir():

            if not model_dir.is_dir():
                continue

            model_name = model_dir.name

            variants = {}

            for p in model_dir.rglob("*"):

                if p.is_dir() and VARIANT_RE.fullmatch(p.name):

                    variants[p.name] = p

            if len(variants) > 0:

                models[model_name] = variants

    return models


def find_files(variant_path, variable, statistic):

    files = []

    var_dir = variant_path / "glb" / "mon" / variable

    if not var_dir.exists():
        return files

    # Find matching directory
    stat_dirs = sorted(
        var_dir.glob(f"{statistic}*")
    )

    for stat_dir in stat_dirs:

        for grid_dir in stat_dir.iterdir():
            if grid_dir.name != "gr1.5":
                continue

            if not grid_dir.is_dir():
                continue

            versions = sorted([
                p for p in grid_dir.iterdir()
                if p.is_dir() and p.name.startswith("v")
            ])

            if len(versions) == 0:
                continue

            # Use newest version only
            latest_version = versions[-1]

            files.extend(
                sorted(
                    latest_version.glob(
                        f"{variable}_*gr1.5*.nc"
                    )
                )
            )

    return sorted(files)


def open_variant(files, variable, syear, eyear, MIN_LAT, MAX_LAT, MIN_LON, MAX_LON):

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=True,
        decode_times=True
    )

    # Subset years
    da = ds[variable]

    da = da.where(
        (da.time.dt.year >= int(syear)) &
        (da.time.dt.year <= int(eyear)),
        drop=True
    )

    # Check year coverage
    expected_years = np.arange(int(syear), int(eyear) + 1)

    actual_years = np.unique(da.time.dt.year.values)

    missing_years = np.setdiff1d(expected_years, actual_years)

    if len(missing_years) > 0:
        raise ValueError(
            f"Missing years after subsetting: "
            f"{missing_years.tolist()}"
        )

    # Spatial subset
    #da = regrid_to_target(da, target_grid)
    da = spatial_subset(da, MIN_LAT, MAX_LAT, MIN_LON, MAX_LON)

    # Seasonal climatology
    da = da.groupby("time.season").mean("time")

    return da

def spatial_subset(da, MIN_LAT, MAX_LAT, MIN_LON, MAX_LON):

    # ----------------------------------------------
    # Convert longitudes to 0-360 if needed
    # ----------------------------------------------

    if da.lon.min() < 0:

        da = da.assign_coords(
            lon=(da.lon % 360)
        )

        da = da.sortby("lon")

    # ----------------------------------------------
    # Latitude subset
    # ----------------------------------------------

    if da.lat[0] > da.lat[-1]:

        da = da.sel(
            lat=slice(MAX_LAT, MIN_LAT)
        )

    else:

        da = da.sel(
            lat=slice(MIN_LAT, MAX_LAT)
        )

    # ----------------------------------------------
    # Longitude subset
    # ----------------------------------------------

    da = da.sel(
        lon=slice(MIN_LON, MAX_LON)
    )

    return da


#%% Set up variables
syr = '1991'
eyr = '2020'

scen = 'historical'

mod_path = '/scratch/qclimdata/cmip7/Data/GCM/'

VARIABLE = "tas"  # variable to process
STATISTIC = 'tavg'

MIN_LAT, MAX_LAT = -45, 10
MIN_LON, MAX_LON = 50, 260

TARGET_FILE='/scratch/qclimdata/Data/Reanalysis/ERA/ERA5/clim/ERA5_gr1.5_2t_t2m_sesmean_' + syr + '-' + eyr + '.nc'

VARIANT_RE = re.compile(r"r\d+i\d+p\d+f\d+")
OUT_DIR = Path("/scratch/qclimdata/cmip7/Data/GCM/model_means")

OUT_DIR.mkdir(parents=True, exist_ok=True)

#%%

# ============================================================
# Main
# ============================================================

target_ds = xr.open_dataset(TARGET_FILE)
target = target_ds.rename(
    {
        "latitude": "lat",
        "longitude": "lon"
    }
)

#target_grid = xr.Dataset({"lat": target_ds.latitude, "lon": target_ds.longitude})

models = find_models(Path(mod_path))

print(f"Found {len(models)} models")

for model_name, variants in sorted(models.items()):

    #if model_name != target_model:
     #   continue

    print()
    print("=" * 80)
    print(model_name)
    print("=" * 80)

    variant_arrays = []
    variant_names = []

    for variant_name, variant_path in sorted(variants.items()):

        files = find_files(variant_path, VARIABLE, STATISTIC)

        if len(files) == 0:
            print(f"Skipping {variant_name} ", "(no files found)")
            continue

        print(f"{variant_name}: ", f"{len(files)} files")

        try:

            da = open_variant(files, VARIABLE, syr, eyr, MIN_LAT, MAX_LAT, MIN_LON, MAX_LON)

            da = da.expand_dims(variant=[variant_name])

            variant_arrays.append(da)
            variant_names.append(variant_name)

        except Exception as e:

            print(f"Failed {variant_name}: {e}")

    if len(variant_arrays) == 0:

        print("No usable variants")
        continue

    print(f"Computing ensemble mean from ", f"{len(variant_arrays)} variants")

    ensemble = xr.concat(variant_arrays, dim="variant")

    model_mean = ensemble.mean(
        dim="variant",
        keep_attrs=True,
        skipna=True
    )

    model_mean.name = VARIABLE

    model_mean.attrs[
        "ensemble_members"
    ] = ",".join(variant_names)

    model_mean.attrs[
        "n_variants"
    ] = len(variant_names)

    outfile = (
        OUT_DIR /
        f"{model_name}_gr1.5_{VARIABLE}_{STATISTIC}_ensemble_mean_{syr}-{eyr}.nc"
    )

    print(f"Writing {outfile}")

    model_mean.to_netcdf(outfile)

    # Calculate bias
    print("Calculating bias")
    bias = model_mean - target

    print("Writing bias")
    bias_outfile = (
    OUT_DIR /
    f"{model_name}_gr1.5_{VARIABLE}_bias_{syr}-{eyr}.nc"
    )

    bias.to_netcdf(bias_outfile)

    print("Done")

print("\nFinished")
# %%
