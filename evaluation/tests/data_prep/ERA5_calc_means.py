#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate seasonal means of ERA5

Sample script to test bias calc and plotting for CMIP6 and CMIP7
-Need to redo with data from lp01

Created on 03/11/2026

Environment: env-fiona-xesmf

@author: schapman
"""
#%%#import packages
import glob
import xarray as xr
from pathlib import Path
import xesmf as xe

#%% Set up variables
syr = '1991'
eyr = '2020'

MIN_LAT, MAX_LAT = -45, 10
MIN_LON, MAX_LON = 50, 260

mod_path = '/scratch/qclimdata/Data/Reanalysis/ERA/ERA5/single-levels/reanalysis/'
VARIABLE = "2t"  # variable to process
STATISTIC = 't2m'

OUT_DIR = Path("/scratch/qclimdata/Data/Reanalysis/ERA/ERA5/clim")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# File for regridding
base_file = '/scratch/qclimdata/cmip7/Data/GCM/ACCESS-Consortium/ACCESS-ESM1-6/historical/r1i1p1f1/glb/mon/tas/tavg-h2m-hxy-u/g115/latest/tas_tavg-h2m-hxy-u_mon_glb_g115_ACCESS-ESM1-6_historical_r1i1p1f1_185001-185912.nc'
dst = xr.open_dataset(base_file)
dst = dst.rename({
    "lat": "latitude",
    "lon": "longitude"
})


#%%
print("Load ERA5")
files = glob.glob(mod_path + VARIABLE + "/*nc")

ds = xr.open_mfdataset(
    files,
    combine="by_coords",
    parallel=True,
    decode_times=True
)

print("Subset years")
# Subset years
ds = ds.sel(time=slice(f"{syr}", f"{eyr}"))

# Regrid
regridder = xe.Regridder(
    ds,
    dst,
    "bilinear",
    periodic=False
)

print("Regridding")
ds_regrid = regridder(ds[STATISTIC])

print("Spatial subset")
da = ds_regrid.sel(latitude=slice(MIN_LAT, MAX_LAT))

if da.longitude.min() < 0:
        da = da.assign_coords(longitude=(da.longitude % 360))
        da = da.sortby("longitude")

da = da.sel(longitude=slice(MIN_LON, MAX_LON))

print("Seasonal climatology")
# Seasonal climatology
da = da.groupby("time.season").mean("time")

print("Saving")
encoding = {
    "latitude": {"_FillValue": None},
    "longitude": {"_FillValue": None},
    "season": {"_FillValue": None},
}

outfile = (OUT_DIR /f"ERA5_{VARIABLE}_{STATISTIC}_sesmean_{syr}-{eyr}.nc")
da.name = STATISTIC
da.to_netcdf(outfile, encoding = encoding)
