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
from run_dataloader import era5_single_level, era5_pressure_level

REF_PERIOD = ("1980-01-01", "2020-01-01")
# REF_PERIOD = ("1950-01-01", "1950-01-15")
TESTING = False
SAVE = True

dataloader = DataLoader(
    root = [
        "/glade/campaign/univ/uwas0118/scratch/archive/1950_2015/",
        "/glade/scratch/zespinosa/archive/cesm2.1.3_BHISTcmip6_f09_g17_ERA5_nudge/", 
        "/glade/scratch/zespinosa/archive/cesm2.1.3_BSSP370cmip6_f09_g17_ERA5_nudge/"
    ],
    era5_root="/glade/work/zespinosa/data/era5/monthly"
)

datatransformer = DataTransformer(
    save_path='/glade/work/zespinosa/Projects/SI-Antarctic/data'
)

def transform_cesm2_data(ds, save, save_name):
    print("starting regrid")
    ds = datatransformer.regrid(
        ds, 
        save=save, 
        save_name=save_name
    )

    print(ds)
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


##### ICE #####
print("starting ice")
ice_cesm2 = dataloader.get_cesm2_data(
    comp="ice",
    myvars=["aice", "daidtt", "daidtd", "dvidtt", "dvidtd", "sithick", "uvel", "vvel"],
    testing=TESTING,
)
transform_cesm2_data(
    ds=ice_cesm2,
    save=SAVE,
    save_name="cesm2_ice_monthly_1950-01_2023-12",
)

##### OCN MXL #####
print("starting ocn mxl")
ocn_mxl = dataloader.get_cesm2_data(comp="ocn", myvars=["HMXL"], testing=TESTING)
transform_cesm2_data(
    ds=ocn_mxl,
    save=SAVE,
    save_name="cesm2_ocn-mxl_monthly_1950-01_2023-12",
)

##### OCN SST #####
print("starting ocn sst")
ocn_sst = dataloader.get_cesm2_data(comp="ocn", myvars=["SST"], testing=TESTING)
transform_cesm2_data(
    ds=ocn_sst,
    save=SAVE,
    save_name="cesm2_ocn-sst_monthly_1950-01_2023-12",
)

##### ATM #####
print("starting atm")
atm_cesm2 = dataloader.get_cesm2_data(
    comp="atm",
    myvars=["PSL", "U10", "TS", "T", "U", "V", "Z3"],
    levels=[1000, 850, 500],
    testing=TESTING
)
transform_cesm2_data(
    ds=atm_cesm2,
    save=SAVE,
    save_name="cesm2_atm_monthly_1950-01_2023-12",
)
