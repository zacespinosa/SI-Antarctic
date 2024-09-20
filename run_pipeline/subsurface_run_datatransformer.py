import os
import sys

from glob import glob
from typing import List, Tuple

import numpy as np
import xarray as xr
import xcdat as xc
import xskillscore as xscore

# Personal Data Loader
sys.path.append('../pipeline')
from data_loader import DataLoader
from data_transformer import DataTransformer
from seaice_transformer import SeaIceTransformer
from run_dataloader import era5_single_level, era5_pressure_level

REF_PERIOD = ("1980-01-01", "2020-01-01")
TESTING = False
SAVE = True

def transform_cesm2_data(datatransformer, ds, save, save_name):
    print("starting regrid")
    ds = datatransformer.regrid(
        ds,
        save=save,
        save_name=save_name
    )

    print("starting anomalies and climatology")
    ds_ac = datatransformer.calculate_anoms_climatology(
        ds,
        ref_period=REF_PERIOD,
        save=save,
        save_name=save_name
    )

    print("starting trends")
    ds_trend = datatransformer.calculate_linear_time_trend(
        ds,
        save=save,
        save_name=save_name
    )
    return ds

def transform_era5_data(datatransformer, era5_data, test, save, cvar):

    if test:
        era5_data = era5_data.sel(time=slice("1979-01-01", "1980-01-01"))

    # Regrid and save
    print("starting regrid")
    era5_data = datatransformer.regrid(ds=era5_data, save=save, save_name=f"ERA5_monthly_1979-01_2023-12_{cvar}")

    # Calculate anomalies and climatology
    print("starting anomalies and climatology")
    era5_data_ac = datatransformer.calculate_anoms_climatology(
        ds=era5_data.copy(),
        ref_period=REF_PERIOD,
        save_name=f"ERA5_monthly_1979-01_2023-12_{cvar}",
        save=save,
    )

    print("starting trends")
    era5_data_trends = datatransformer.calculate_linear_time_trend(
        ds=era5_data.copy(),
        save=True,
        save_name=f"ERA5_monthly_1979-01_2023-12_{cvar}",
    )

dataloader = DataLoader(
    root = [
        "/glade/campaign/univ/uwas0118/scratch/archive/1950_2015/",
        "/glade/derecho/scratch/zespinosa/archive/cesm2.1.3_BHISTcmip6_f09_g17_ERA5_nudge/",
        "/glade/derecho/scratch/zespinosa/archive/cesm2.1.3_BSSP370cmip6_f09_g17_ERA5_nudge/"
    ],
    era5_root="/glade/work/zespinosa/data/era5/monthly"
)

print("starting ocn vars")
myvars = ["SALT", "TEMP", "RHO", "HMXL"]
ocn_mxl = dataloader.get_cesm2_data(comp="ocn", myvars=myvars, testing=TESTING)
print(ocn_mxl)
datatransformer = DataTransformer(save_path='/glade/work/zespinosa/Projects/SI-Antarctic/data/')
ds = transform_cesm2_data(
    datatransformer=datatransformer,
    ds=ocn_mxl,
    save=SAVE,
    save_name="cesm2_ocn-subsurface_monthly_1950-01_2023-12",
)