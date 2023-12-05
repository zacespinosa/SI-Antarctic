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

def transform_cesm2_data(datatransformer, ds, save, save_name):
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



##### ERA5 #####
# Load data
# era5_single = era5_single_level(dataloader)
# era5_pressure = era5_pressure_level(dataloader)

# era5_single = era5_single.isel(expver=0).squeeze()
# era5_pressure = era5_pressure.isel(expver=0).squeeze()

# era5_single = era5_single.drop('expver')
# era5_pressure = era5_pressure.drop('expver')
# print(era5_single)
# print(era5_pressure)


# # Start Transform
# print("Starting single level")
# transform_era5_data(era5_single, test=False, save=True, cvar="full-single")
# print("Starting pressure level")
# transform_era5_data(era5_pressure, test=False, save=True, cvar="full-pressure")
########################## CESM2 Persistence Ensemble ####################################
def transform_cesm2_ens():
    dataloader = DataLoader(
        root = [
            "/glade/campaign/univ/uwas0118/scratch/archive/1950_2015/",
            "/glade/scratch/zespinosa/archive/cesm2.1.3_BHISTcmip6_f09_g17_ERA5_nudge/",
            "/glade/scratch/zespinosa/archive/cesm2.1.3_BSSP370cmip6_f09_g17_ERA5_nudge/"
        ],
        era5_root="/glade/work/zespinosa/data/era5/monthly"
    )

    members = ["1980", "1989", "1994", "1998", "1999", "2005", "2007", "2014", "2015", "2018"]
    for ens_mem in members:
        print("Starting member: ", ens_mem)
        dataloader_ens = DataLoader(
            root = [
                f"/glade/scratch/zespinosa/archive/{ens_mem}_cesm2.1.3_BSSP370cmip6_f09_g17_ERA5_nudge"
            ],
            era5_root="/glade/work/zespinosa/data/era5/monthly"
        )

        datatransformer = DataTransformer(
            save_path=f'/glade/work/zespinosa/Projects/SI-Antarctic/data/persistence_ensemble/{ens_mem}',
        )

        ##### ICE #####
        print("starting ice")
        ice_cesm2 = dataloader.get_cesm2_data(
            comp="ice",
            myvars=["aice", "daidtt", "daidtd", "dvidtt", "dvidtd", "sithick", "uvel", "vvel"],
            testing=TESTING,
        )
        ice_cesm2_enso = dataloader_ens.get_cesm2_data(
            comp="ice",
            myvars=["aice", "daidtt", "daidtd", "dvidtt", "dvidtd", "sithick", "uvel", "vvel"],
            testing=TESTING,
        )
        # Merge ENSO and NO ENSO
        ice_cesm2 = xr.concat([
            ice_cesm2.sel(time=slice("1950-01-15", "2022-12-15")),
            ice_cesm2_enso,
        ], dim="time")

        transform_cesm2_data(
            datatransformer=datatransformer,
            ds=ice_cesm2,
            save=SAVE,
            save_name=f"{ens_mem}_cesm2_ice_monthly_1950-01_2023-12",
        )

        ##### OCN SST #####
        print("starting ocn sst")
        ocn_sst = dataloader.get_cesm2_data(comp="ocn", myvars=["SST"], testing=TESTING)
        ocn_sst_enso = dataloader_ens.get_cesm2_data(comp="ocn", myvars=["SST"], testing=TESTING)
        # Merge ENSO and NO ENSO
        ocn_sst = xr.concat([
            ocn_sst.sel(time=slice("1950-01-15", "2022-12-15")),
            ocn_sst_enso,
        ], dim="time")

        transform_cesm2_data(
            datatransformer=datatransformer,
            ds=ocn_sst,
            save=SAVE,
            save_name=f"{ens_mem}_cesm2_ocn-sst_monthly_1950-01_2023-12",
        )

        ##### ATM #####
        print("starting atm")
        atm_cesm2 = dataloader.get_cesm2_data(
            comp="atm",
            myvars=["PSL", "U10", "TS", "T", "U", "V", "Z3"],
            levels=[1000, 850, 500],
            testing=TESTING
        )
        atm_cesm2_enso = dataloader_ens.get_cesm2_data(
            comp="atm",
            myvars=["PSL", "U10", "TS", "T", "U", "V", "Z3"],
            levels=[1000, 850, 500],
            testing=TESTING
        )
        # Merge ENSO and NO ENSO
        atm_cesm2 = xr.concat([
            atm_cesm2.sel(time=slice("1950-01-15", "2022-12-15")),
            atm_cesm2_enso,
        ], dim="time")

        transform_cesm2_data(
            datatransformer=datatransformer,
            ds=atm_cesm2,
            save=SAVE,
            save_name=f"{ens_mem}_cesm2_atm_monthly_1950-01_2023-12",
        )

transform_cesm2_ens()


########################## CESM2 NO ENSO ####################################
# dataloader_enso = DataLoader(
#     root = [
#         "/glade/scratch/zespinosa/archive/NO_ENSO_cesm2.1.3_BSSP370cmip6_f09_g17_ERA5_nudge"
#     ],
#     era5_root="/glade/work/zespinosa/data/era5/monthly"
# )

# ##### ICE #####
# print("starting ice")
# ice_cesm2 = dataloader.get_cesm2_data(
#     comp="ice",
#     myvars=["aice", "daidtt", "daidtd", "dvidtt", "dvidtd", "sithick", "uvel", "vvel"],
#     testing=TESTING,
# )
# ice_cesm2_enso = dataloader_enso.get_cesm2_data(
#     comp="ice",
#     myvars=["aice", "daidtt", "daidtd", "dvidtt", "dvidtd", "sithick", "uvel", "vvel"],
#     testing=TESTING,
# )
# # Merge ENSO and NO ENSO
# ice_cesm2 = xr.concat([
#     ice_cesm2.sel(time=slice("1950-01-15", "2022-12-15")),
#     ice_cesm2_enso,
# ], dim="time")

# transform_cesm2_data(
#     ds=ice_cesm2,
#     save=SAVE,
#     save_name="enso_cesm2_ice_monthly_1950-01_2023-12",
# )

##### OCN MXL #####
# print("starting ocn mxl")
# ocn_mxl = dataloader.get_cesm2_data(comp="ocn", myvars=["HMXL"], testing=TESTING)
# transform_cesm2_data(
#     ds=ocn_mxl,
#     save=SAVE,
#     save_name="cesm2_ocn-mxl_monthly_1950-01_2023-12",
# )

# ##### OCN SST #####
# print("starting ocn sst")
# ocn_sst = dataloader.get_cesm2_data(comp="ocn", myvars=["SST"], testing=TESTING)
# ocn_sst_enso = dataloader_enso.get_cesm2_data(comp="ocn", myvars=["SST"], testing=TESTING)
# # Merge ENSO and NO ENSO
# ocn_sst = xr.concat([
#     ocn_sst.sel(time=slice("1950-01-15", "2022-12-15")),
#     ocn_sst_enso,
# ], dim="time")

# transform_cesm2_data(
#     ds=ocn_sst,
#     save=SAVE,
#     save_name="enso_cesm2_ocn-sst_monthly_1950-01_2023-12",
# )

##### ATM #####
# print("starting atm")
# atm_cesm2 = dataloader.get_cesm2_data(
#     comp="atm",
#     myvars=["PSL", "U10", "TS", "T", "U", "V", "Z3"],
#     levels=[1000, 850, 500],
#     testing=TESTING
# )
# atm_cesm2_enso = dataloader_enso.get_cesm2_data(
#     comp="atm",
#     myvars=["PSL", "U10", "TS", "T", "U", "V", "Z3"],
#     levels=[1000, 850, 500],
#     testing=TESTING
# )
# # Merge ENSO and NO ENSO
# atm_cesm2 = xr.concat([
#     atm_cesm2.sel(time=slice("1950-01-15", "2022-12-15")),
#     atm_cesm2_enso,
# ], dim="time")

# transform_cesm2_data(
#     ds=atm_cesm2,
#     save=SAVE,
#     save_name="enso_cesm2_atm_monthly_1950-01_2023-12",
# )


########################## CESM2 ####################################
##### ICE #####
# print("starting ice")
# ice_cesm2 = dataloader.get_cesm2_data(
#     comp="ice",
#     myvars=["aice", "daidtt", "daidtd", "dvidtt", "dvidtd", "sithick", "uvel", "vvel"],
#     testing=TESTING,
# )
# transform_cesm2_data(
#     ds=ice_cesm2,
#     save=SAVE,
#     save_name="cesm2_ice_monthly_1950-01_2023-12",
# )

##### OCN MXL #####
# print("starting ocn mxl")
# ocn_mxl = dataloader.get_cesm2_data(comp="ocn", myvars=["HMXL"], testing=TESTING)
# transform_cesm2_data(
#     ds=ocn_mxl,
#     save=SAVE,
#     save_name="cesm2_ocn-mxl_monthly_1950-01_2023-12",
# )

##### OCN SST #####
# print("starting ocn sst")
# ocn_sst = dataloader.get_cesm2_data(comp="ocn", myvars=["SST"], testing=TESTING)
# transform_cesm2_data(
#     ds=ocn_sst,
#     save=SAVE,
#     save_name="cesm2_ocn-sst_monthly_1950-01_2023-12",
# )

##### ATM #####
# print("starting atm")
# atm_cesm2 = dataloader.get_cesm2_data(
#     comp="atm",
#     myvars=["PSL", "U10", "TS", "T", "U", "V", "Z3"],
#     levels=[1000, 850, 500],
#     testing=TESTING
# )
# transform_cesm2_data(
#     ds=atm_cesm2,
#     save=SAVE,
#     save_name="cesm2_atm_monthly_1950-01_2023-12",
# )
