#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare climatology of CMIP6 and CMIP7 ensemble means

Test plotting script
- need to redo with lp01 data
- need to check and improve stippling

Created on 4/11/2026

@author: schapman
"""
#%%#import packages
import numpy as np
import glob
import copy
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import cartopy.feature as cfeature
import sys
sys.path.append('/export/home/qclim/repositories/common-functions/')
import plotting_functions as tp 

import iris
import iris.cube
import iris.plot as iplt

plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.ion() #turns off figure plotting plt.ion() to get it back on

fs = 8
lw = 1.0

matplotlib.rcParams.update({'font.size': fs})

proj = ccrs.PlateCarree(central_longitude=180)

#%% Load data

var = 'tas'
file_var = 't2m'
syr = '1991'
eyr = '2020'

#% Load CMIP6
cmip6_path = '/scratch/qclimdata/Data/GCM/model_means/'
cmip7_path = '/scratch/qclimdata/cmip7/Data/GCM/model_means/'

save_path = '/scratch/qclimprod/CMIP7/Papers/gcm_evaluation/Figures/'

#%% #Load model data
def calc_ens(fpath, syr, eyr, var, file_var):

    file_names = glob.glob(fpath + '*bias_' + syr + '-' + eyr + '.nc')
    file_names = [x for x in file_names if var in x]

    mod_list = []

    for file in file_names:

        x_xr = xr.load_dataset(file)

        ens = file.partition('model_means/')[2].partition("_")[0]

        mod_list.append(x_xr[file_var])

    # dimensions: ens, season, lat, lon
    ens_dat = xr.concat(
        mod_list,
        dim='ens',
        coords='minimal',
        compat='override'
    )

    ens_mean = ens_dat.mean('ens')

	# This part needs to be checked and improved
    nmods = ens_dat.sizes['ens']
    threshold = np.ceil(2 * nmods / 3)

    pos_count = (ens_dat > 0).sum('ens')
    neg_count = (ens_dat < 0).sum('ens')

    agreement = (pos_count >= threshold) | (neg_count >= threshold)

    return ens_mean, agreement

cmip6_mean, cmip6_agree = calc_ens(cmip6_path, syr, eyr, var, file_var)
cmip7_mean, cmip7_agree = calc_ens(cmip7_path, syr, eyr, var, file_var)

#%% Ens avg BIAS plots

print("Plotting...")
plot_list = [cmip6_mean.sel(season = 'DJF'), cmip7_mean.sel(season = 'DJF'), 
             cmip6_mean.sel(season = 'MAM'), cmip7_mean.sel(season = 'MAM'), 
             cmip6_mean.sel(season = 'JJA'), cmip7_mean.sel(season = 'JJA'), 
             cmip6_mean.sel(season = 'SON'), cmip7_mean.sel(season = 'SON')]

agree_list = [cmip6_agree.sel(season = 'DJF'), cmip7_agree.sel(season = 'DJF'), 
             cmip6_agree.sel(season = 'MAM'), cmip7_agree.sel(season = 'MAM'), 
             cmip6_agree.sel(season = 'JJA'), cmip7_agree.sel(season = 'JJA'), 
             cmip6_agree.sel(season = 'SON'), cmip7_agree.sel(season = 'SON')]


fig = plt.figure(figsize = (8,8))
rows = 4
cols = 2
nfigs = rows * cols

no_grids = np.arange(1,nfigs - cols)[np.mod(np.arange(1, nfigs - cols),cols)!= 0]
y_only = np.arange(0, nfigs - cols, cols)

lev_tas = np.arange(-5, 5.5, 0.5)
col_b = 'RdBu_r'
skip = 4
s=0.5
alpha=0.8

labels = ['a1', 'b1',
          'a2', 'b2',
          'a3', 'b3',
          'a4', 'b4']

ax_list = []
for i in np.arange(1, nfigs+1):
    ax_list.append(plt.subplot(rows, cols, i, projection = proj))

for i in np.arange(len(ax_list)):
    
    ax = ax_list[i]
    ax.set_extent([50, 260, -45, 10], crs=ccrs.PlateCarree())
    cube = plot_list[i].to_iris()

    im_tas = iplt.contourf(cube, axes = ax, levels = lev_tas, cmap = col_b, extend = 'both')

    xpos, ypos = 0.01, 0.92
    ax.annotate(text = labels[i], xy = (xpos, ypos), xycoords = 'axes fraction',
                                ha = 'left', fontsize = fs, weight = 'bold')

    # Add stippling mask
    agree = agree_list[i]

    lon2d, lat2d = np.meshgrid(
        agree.longitude,
        agree.latitude
    )

    mask = agree.values

    ax.scatter(
        lon2d[::skip, ::skip][mask[::skip, ::skip]],
        lat2d[::skip, ::skip][mask[::skip, ::skip]],
        s=s,
        alpha = alpha,
        c='k',
        marker='.',
        transform=ccrs.PlateCarree(),
        zorder=5
    )

    if i in no_grids:
         print('no grids')
    elif i in y_only:
        ax.set_yticks([-40,-20,0], crs=ccrs.PlateCarree())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.set_xticklabels([])
    elif i == no_grids[-1] + 1:
        ax.set_yticks([-40,-20,0], crs=ccrs.PlateCarree())
        ax.set_xticks([60,120,180,240], crs=ccrs.PlateCarree())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))
    else:
        ax.set_xticks([60,120,180,240], crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))
        ax.set_yticklabels([])
  
    ax.set_aspect('auto')

    ax.coastlines(resolution='50m',
              linewidth=0.6,
              color='black')

    ax.add_feature(
    cfeature.LAND,
    facecolor='lightgrey',
    edgecolor='none',
    zorder=0
    )

    gl = ax.gridlines(
    draw_labels=False,
    linewidth=0.25,
    color='0.7',
    linestyle='--'
    ) 


xpos = 0.5
ypos = 1.05
ax_list[0].annotate(text = 'CMIP6', xy = (xpos, ypos), xycoords = 'axes fraction', ha = 'center', weight='bold')	     
ax_list[1].annotate(text = 'CMIP7', xy = (xpos, ypos), xycoords = 'axes fraction', ha = 'center', weight='bold')

xpos = -0.25
ypos = 0.5
ax_list[0].annotate(text = 'DJF', xy = (xpos, ypos), xycoords = 'axes fraction',
                va = 'center', rotation = 90, fontsize = 10, weight = 'bold')
ax_list[2].annotate(text = 'MAM', xy = (xpos, ypos), xycoords = 'axes fraction',
                va = 'center', rotation = 90, fontsize = 10, weight = 'bold')
ax_list[4].annotate(text = 'JJA', xy = (xpos, ypos), xycoords = 'axes fraction',
                va = 'center', rotation = 90, fontsize = 10, weight = 'bold')
ax_list[6].annotate(text = 'SON', xy = (xpos, ypos), xycoords = 'axes fraction',
                va = 'center', rotation = 90, fontsize = 10, weight = 'bold')

plt.draw()

dif = 0.04
hw = 0.01


#BIAS
cbax = tp.get_cbax(fig, ax = ax_list[6], last_ax = [ax_list[5]], orientation = 'horizontal', dif = dif, h_w = hw)
cbar = plt.colorbar(im_tas, cax = cbax, orientation = 'horizontal')
cbar.set_label('tas bias (°C)')

plt.subplots_adjust(
    hspace=0.05,
    wspace=0.02
)


#save fig
fig.savefig(save_path + 'BIAS_ERA5_ModAvg_' + var + '_' + syr + '-' + eyr + '.jpg',
                            bbox_inches = 'tight', pad_inches = 0.3)      



# %%
