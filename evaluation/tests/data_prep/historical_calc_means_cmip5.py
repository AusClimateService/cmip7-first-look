#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate seasonal means of tas and other vars over various historical periods - CMIP6 for comparison

# will need historical + future scenario to get full time series

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

def find_models(base_dir, scen="historical"):

    models = {}

    for model_dir in base_dir.iterdir():

        if not model_dir.is_dir():
            continue

        amon_dir = (
            model_dir /
            scen /
            "mon" /
            "atmos" /
            "Amon"
        )

        if not amon_dir.exists():
            continue

        variants = {}

        for variant_dir in amon_dir.iterdir():

            if (
                variant_dir.is_dir()
                and VARIANT_RE.fullmatch(variant_dir.name)
            ):
                variants[variant_dir.name] = variant_dir

        if variants:
            models[model_dir.name] = variants

    return models

def find_files(variant_path, variable):

    files = []

    version_dirs = sorted([
        p for p in variant_path.iterdir()
        if p.is_dir() and p.name.startswith("v")
    ])

    if len(version_dirs) == 0:
        return files

    latest_version = version_dirs[-1]

    var_dir = latest_version / variable

    if not var_dir.exists():
        return files

    files.extend(
        sorted(var_dir.glob("*gr1.5*.nc"))
    )

    return files

def get_variant_files(hist_variant_path, model_name, variable, ssp_scen):

    future_variant = (
        Path(mod_path)
        / model_name
        / ssp_scen
        / "mon"
        / "atmos"
        / "Amon"
        / hist_variant_path.name
    )

    if not hist_variant_path.exists():
        return []

    if not future_variant.exists():
        return []

    hist_files = find_files(
        hist_variant_path,
        variable
    )

    ssp_files = find_files(
        future_variant,
        variable
    )

    return sorted(hist_files + ssp_files)


def open_variant(files, VARIABLE, syear, eyear, MIN_LAT, MAX_LAT, MIN_LON, MAX_LON):

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=True,
        decode_times=True
    )

    # Subset years
    ds = ds.where(
        (ds.time.dt.year >= int(syear)) &
        (ds.time.dt.year <= int(eyear)),
        drop=True
    )       

    # Check all years present in combined historical + ssp data
    expected_years = np.arange(int(syear),int(eyear) + 1)

    actual_years = np.unique(
        ds.time.dt.year.values
    )

    missing_years = np.setdiff1d(
        expected_years,
        actual_years
    )

    if len(missing_years) > 0:
        raise ValueError(
            f"Missing years: "
            f"{missing_years.tolist()}"
        )

    # Spatial subset
    da = spatial_subset(ds[VARIABLE], MIN_LAT, MAX_LAT, MIN_LON, MAX_LON)

    #da = regrid_to_target(ds[variable], target_grid)

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
ssp_scen = 'rcp45' # which scenario to concatenate with historical to get post 2014 data

mod_path = '/scratch/qclimdata/cmip5/Data/GCM/'

VARIABLE = "tas"  # variable to process

MIN_LAT, MAX_LAT = -45, 10
MIN_LON, MAX_LON = 50, 260

TARGET_FILE='/scratch/qclimdata/Data/Reanalysis/ERA/ERA5/clim/ERA5_gr1.5_2t_t2m_sesmean_' + syr + '-' + eyr + '.nc'

VARIANT_RE = re.compile(r"r\d+i\d+p\d+")
OUT_DIR = Path("/scratch/qclimdata/cmip5/Data/GCM/model_means")

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

target_grid = xr.Dataset(
    {
        "lat": target_ds.latitude,
        "lon": target_ds.longitude,
    }
)

models = find_models(Path(mod_path), scen)

print(f"Found {len(models)} models")

for model_name, variants in sorted(models.items()):

    print()
    print("=" * 80)
    print(model_name)
    print("=" * 80)

    variant_arrays = []
    variant_names = []

    for variant_name, variant_path in sorted(variants.items()):

        files = get_variant_files(variant_path, model_name, VARIABLE, ssp_scen)

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
        f"{model_name}_gr1.5_{VARIABLE}_ensemble_mean_historical+{ssp_scen}_{syr}-{eyr}.nc"
    )

    print(f"Writing {outfile}")

    model_mean.to_netcdf(outfile)

    print(np.nanmean(model_mean[0]) - 273.15)

    # Calculate bias
    print("Calculating bias")
    bias = model_mean - target

    print("Writing bias")
    bias_outfile = (
    OUT_DIR /
    f"{model_name}_gr1.5_{VARIABLE}_bias_historical+{ssp_scen}_{syr}-{eyr}.nc"
    )

    bias.to_netcdf(bias_outfile)

    print("Done")

print("\nFinished")
# %%
