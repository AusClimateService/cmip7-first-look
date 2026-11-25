#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate seasonal means of tas and other vars over various historical periods - CMIP5 for comparison

# will need historical + future scenario to get full time series
# will check for corrected files and use if in range from syear - eyear

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

def find_files(variant_path, variable):

    files = []

    var_dir = variant_path / variable

    if not var_dir.exists():
        return files

    versions = sorted([
        p for p in var_dir.iterdir()
        if p.is_dir() and p.name.startswith("v")
    ])

    if len(versions) == 0:
        return files

    latest_version = versions[-1]

    files.extend(
        sorted(latest_version.glob("*.nc"))
    )

    return files

def find_models(base_dir,
                hist_scen="historical",
                future_scen="sresa1b"):

    models = {}

    for model_dir in base_dir.iterdir():

        if not model_dir.is_dir():
            continue

        hist_dir = (
            model_dir / hist_scen /
            "mon" / "atmos"
        )

        fut_dir = (
            model_dir / future_scen /
            "mon" / "atmos"
        )

        if not hist_dir.exists():
            continue

        if not fut_dir.exists():
            continue

        hist_runs = {
            p.name: p
            for p in hist_dir.iterdir()
            if p.is_dir() and p.name.startswith("run")
        }

        fut_runs = {
            p.name: p
            for p in fut_dir.iterdir()
            if p.is_dir() and p.name.startswith("run")
        }

        common_runs = (
            hist_runs.keys()
            & fut_runs.keys()
        )

        if common_runs:

            models[model_dir.name] = {
                run: hist_runs[run]
                for run in common_runs
            }

    return models

def get_variant_files(hist_variant_path, model_name, variable, future_scen):

    hist_files = find_files(hist_variant_path, variable)

    if len(hist_files) == 0:
        return []

    # check historical coverage (20C3M files go into future period)
    hist_ds = xr.open_dataset(
        hist_files[0],
        decode_times=True,
        use_cftime=True
    )

    hist_end_year = int(
        hist_ds.time.dt.year.max()
    )

    hist_ds.close()

    if hist_end_year >= int(eyr):

        print(
            f"{model_name} {hist_variant_path.name}: "
            f"historical already reaches "
            f"{hist_end_year}"
        )

        return hist_files

    future_variant = (
        Path(mod_path)
        / model_name
        / future_scen
        / "mon"
        / "atmos"
        / hist_variant_path.name
    )

    if future_variant.exists():
        fut_files = find_files(
            future_variant,
            variable
        )
    else:
        fut_files = []

    # --------------------------------------------------
    # Detect overlapping corrected files
    # --------------------------------------------------

    selected_future = []

    for f in fut_files:

        # normal file
        if ".corrected." not in f.name:
            selected_future.append(f)
            continue

        ds_corr = xr.open_dataset(
            f,
            decode_times=True,
            use_cftime=True
        )

        start_year = int(
            ds_corr.time.dt.year.min()
        )

        end_year = int(
            ds_corr.time.dt.year.max()
        )

        ds_corr.close()

        # only keep corrected files
        # if they overlap analysis years
        if (
            start_year <= int(eyr)
            and end_year >= int(syr)
        ):

            print(
                f"Including correction: "
                f"{f.name} "
                f"({start_year}-{end_year})"
            )

            selected_future.append(f)

    # remove duplicated filenames only
    all_files = sorted(
        {str(f): f for f in hist_files + selected_future}.values(),
        key=str
    )

    return all_files



def open_variant(files, VARIABLE, syear, eyear, MIN_LAT, MAX_LAT, MIN_LON, MAX_LON):

    # Open files and check for overlaps
    datasets = [xr.open_dataset(f, decode_times=True, use_cftime=True)
                    for f in files]

    ds = xr.concat(
        datasets,
        dim="time"
    )

    ds = ds.sortby("time")

    _, index = np.unique(
        ds.time.values,
        return_index=True
    )

    # Remove duplicate times (some of the future files are overlapping)
    ds = ds.isel(time=index)

#    ds = xr.open_mfdataset(
 #       files,
  #      combine="by_coords",
  #      parallel=True,
   #     decode_times=True,
   #     use_cftime=True
   # )

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
    #da = spatial_subset(ds[VARIABLE], MIN_LAT, MAX_LAT, MIN_LON, MAX_LON)
    rename_dict = {}

    if "latitude" in ds.dims:
        rename_dict["latitude"] = "lat"

    if "longitude" in ds.dims:
        rename_dict["longitude"] = "lon"

    if rename_dict:
        ds = ds.rename(rename_dict)

    try:
        da = regrid_to_target(ds[VARIABLE], target_grid)
    except ValueError as e:
        # add exception for some models with essentially duplicated lat values
        
        if "ESMC_FieldRegridStore" not in str(e):
            raise

        da0 = ds[VARIABLE]
        lat = da0.lat.values

        # remove coords where the difference is small
        keep = np.insert(
            np.abs(np.diff(lat)) > 1e-10,
            0,
            True
        )

        da0 = da0.isel(lat=keep)

        print(
        f"Removed {len(lat)-len(da0.lat)} "
        "near-duplicate latitudes"
        )
        da = regrid_to_target(da0, target_grid)

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
ssp_scen = 'sresa1b' # which scenario to concatenate with historical to get post 2014 data

mod_path = '/scratch/qclimdata/cmip3/Data/GCM/'

VARIABLE = "tas"  # variable to process

MIN_LAT, MAX_LAT = -45, 10
MIN_LON, MAX_LON = 50, 260

TARGET_FILE='/scratch/qclimdata/Data/Reanalysis/ERA/ERA5/clim/ERA5_gr1.5_2t_t2m_sesmean_' + syr + '-' + eyr + '.nc'

OUT_DIR = Path("/scratch/qclimdata/cmip3/Data/GCM/model_means")

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
        "lat": target.lat,
        "lon": target.lon,
    }
)

models = find_models(Path(mod_path), hist_scen = scen, future_scen=ssp_scen)

print(f"Found {len(models)} models")

for model_name, variants in sorted(models.items()):

    if model_name == "ipsl_cm4":
        break

    print()
    print("=" * 80)
    print(model_name)
    print("=" * 80)

    variant_arrays = []
    variant_names = []

    for variant_name, variant_path in sorted(variants.items()):

        #if variant_name == "run3":
        #    break

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
