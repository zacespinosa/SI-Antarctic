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



########################## CESM2 Forecast Ensemble ####################################
def transform_cesm2_ens():
    dataloader = DataLoader(
        root = [
            "/glade/campaign/univ/uwas0118/scratch/archive/1950_2015/",
            "/glade/derecho/scratch/zespinosa/archive/cesm2.1.3_BHISTcmip6_f09_g17_ERA5_nudge/",
            "/glade/derecho/scratch/zespinosa/archive/cesm2.1.3_BSSP370cmip6_f09_g17_ERA5_nudge/"
        ],
        era5_root="/glade/work/zespinosa/data/era5/monthly"
    )
    dataloader_cont = DataLoader(
        root = [
            "/glade/derecho/scratch/zespinosa/archive/cont_cesm2.1.3_BSSP370cmip6_f09_g17_ERA5_nudge/"
        ],
        era5_root="/glade/work/zespinosa/data/era5/monthly"
    )

    # members = ["1980", "1985", "1989", "1990", "1993", "1994", "1998"]
    # members = ["1999", "2000", "2003", "2004", "2005", "2006", "2007", "2009"]
    # members = ["2012", "2013", "2014", "2016", "2018", "2020", "2021"]
    members = ["2023"]

    for ens_mem in members:
        print("Starting member: ", ens_mem)
        dataloader_ens = DataLoader(
            root = [
                f"/glade/derecho/scratch/zespinosa/archive/forecast_{ens_mem}_cesm2.1.3_BSSP370cmip6_f09_g17_ERA5_nudge"
            ],
            era5_root="/glade/work/zespinosa/data/era5/monthly"
        )

        datatransformer = DataTransformer(
            # save_path=f'/glade/work/zespinosa/Projects/SI-Antarctic/data/forecast_ensemble/{ens_mem}',
            save_path="/glade/derecho/scratch/zespinosa/SI-Antarctic/data",
        )

        # cice_transformer = SeaIceTransformer(
        #     save_path=f'/glade/work/zespinosa/Projects/SI-Antarctic/data/forecast_ensemble/{ens_mem}',
        # )

        # process_seaice(
        #     dataloader=dataloader,
        #     dataloader_cont=dataloader_cont,
        #     dataloader_ens=dataloader_ens,
        #     datatransformer=datatransformer,
        #     cice_transformer=cice_transformer,
        #     ens_mem=ens_mem
        # )

        process_ocean(
            dataloader=dataloader,
            dataloader_cont=dataloader_cont,
            dataloader_ens=dataloader_ens,
            datatransformer=datatransformer,
            ens_mem=ens_mem
        )

        process_atm(
            dataloader=dataloader,
            dataloader_cont=dataloader_cont,
            dataloader_ens=dataloader_ens,
            datatransformer=datatransformer,
            ens_mem=ens_mem
        )


##### ICE #####
def process_seaice(dataloader, dataloader_cont, dataloader_ens, datatransformer, cice_transformer, ens_mem):
    """
    Process sea ice data from ensemble member: 
    Calculate SIA/SIE and Regions. Calc siconc, siconc trend, anomalies
    """
    print("starting ice")
    ice_cesm2_ens = dataloader_ens.get_cesm2_data(
        comp="ice",
        myvars=["aice", "daidtt", "daidtd", "dvidtt", "dvidtd", "sithick", "uvel", "vvel"],
        testing=TESTING,
    )

    ice_cesm2_cont = dataloader_cont.get_cesm2_data(
        comp="ice",
        myvars=["aice", "daidtt", "daidtd", "dvidtt", "dvidtd", "sithick", "uvel", "vvel"],
        testing=TESTING,
    )

    ice_cesm2 = dataloader.get_cesm2_data(
        comp="ice",
        myvars=["aice", "daidtt", "daidtd", "dvidtt", "dvidtd", "sithick", "uvel", "vvel"],
        testing=TESTING,
    )

    # Merge ENS member
    ice_cesm2 = xr.concat([
        ice_cesm2.sel(time=slice("1950-01-15", "2022-12-15")),
        ice_cesm2_cont.sel(time=slice("2023-01-15", "2023-12-15")),
        ice_cesm2_ens.sel(time=slice("2024-01-15", "2024-12-15")), # Forecast
    ], dim="time")

    # Data is regrided in transform_cesm2_data
    ice_cesm2 = transform_cesm2_data(
        datatransformer=datatransformer,
        ds=ice_cesm2,
        save=SAVE,
        save_name=f"{ens_mem}_cesm2_ice_monthly_1950-01_2024-12",
    )

    # Get Grid Cell Area
    areacello = datatransformer.get_grid_cell_area(ice_cesm2)
    # Regions
    si_cesm2_regions, si_cesm2_regions_anoms = cice_transformer.calc_regions(ice_cesm2["aice"], areacello, prod="CESM", polar=False, save=SAVE)
    # Raw CESM2
    si_cesm2, si_cesm2_anoms = cice_transformer.calc_sia_sie(ice_cesm2["aice"], area=areacello, hem="SH", prod="CESM", save=SAVE)

    print(" - finished ice")

def process_ocean(dataloader, dataloader_cont,  dataloader_ens, datatransformer, ens_mem):
    """
    Process sst data from ensemble member: 
    Calc trend, anomalies, climatology
    """
    print("starting ocn sst")
    myvars = ["SALT", "TEMP", "RHO", "HMXL"]
    # myvars = ["SST"]
    ocn_sst = dataloader.get_cesm2_data(comp="ocn", myvars=myvars, testing=TESTING)
    ocn_sst_cont = dataloader_cont.get_cesm2_data(comp="ocn", myvars=myvars, testing=TESTING)
    ocn_sst_ens = dataloader_ens.get_cesm2_data(comp="ocn", myvars=myvars, testing=TESTING)
    # Merge Historical and New Members
    ocn_sst = xr.concat([
        ocn_sst.sel(time=slice("1950-01-15", "2022-12-15")),
        ocn_sst_cont.sel(time=slice("2023-01-15", "2023-12-15")),
        ocn_sst_ens,
    ], dim="time")
    ocn_sst['lat'] = ocn_sst_ens['lat']
    ocn_sst['lon'] = ocn_sst_ens['lon']


    _ = transform_cesm2_data(
        datatransformer=datatransformer,
        ds=ocn_sst,
        save=SAVE,
        save_name=f"{ens_mem}_cesm2_ocn-subsurface_monthly_1950-01_2024-12",
    )
    print(" - finished ocn")

def process_atm(dataloader, dataloader_cont, dataloader_ens, datatransformer, ens_mem):
    """
    Process atm data from ensemble member: 
    Calc trends, anomalies, climatology
    """
    print("starting atm")
    atm_cesm2 = dataloader.get_cesm2_data(
        comp="atm",
        myvars=["PSL", "U10", "TS", "T", "U", "V", "Z3", "QFLX", "PRECC", "PRECL"],
        levels=[1000, 850, 500],
        testing=TESTING
    )
    print(atm_cesm2)
    atm_cesm2_cont = dataloader_cont.get_cesm2_data(
        comp="atm",
        myvars=["PSL", "U10", "TS", "T", "U", "V", "Z3", "QFLX", "PRECC", "PRECL"],
        levels=[1000, 850, 500],
        testing=TESTING
    )
    print(atm_cesm2_cont)
    atm_cesm2_ens = dataloader_ens.get_cesm2_data(
        comp="atm",
        myvars=["PSL", "U10", "TS", "T", "U", "V", "Z3", "QFLX", "PRECC", "PRECL"],
        levels=[1000, 850, 500],
        testing=TESTING
    )
    print(atm_cesm2_ens)

    atm_cesm2 = xr.concat([
        atm_cesm2.sel(time=slice("1950-01-15", "2022-12-15")),
        atm_cesm2_cont.sel(time=slice("2023-01-15", "2023-12-15")),
        atm_cesm2_ens,
    ], dim="time")

    _ = transform_cesm2_data(
        datatransformer=datatransformer,
        ds=atm_cesm2,
        save=SAVE,
        save_name=f"{ens_mem}_cesm2_atm_monthly_1950-01_2024-12",
    )
    print(" - finished atm")

# transform_cesm2_ens()

# dataloader = DataLoader(
#     root = [
#         "/glade/campaign/univ/uwas0118/scratch/archive/1950_2015/",
#         "/glade/derecho/scratch/zespinosa/archive/cesm2.1.3_BHISTcmip6_f09_g17_ERA5_nudge/",
#         "/glade/derecho/scratch/zespinosa/archive/cesm2.1.3_BSSP370cmip6_f09_g17_ERA5_nudge/"
#     ],
#     era5_root="/glade/work/zespinosa/data/era5/monthly"
# )

# print("starting ocn mxl")
# ocn_mxl = dataloader.get_cesm2_data(comp="ocn", myvars=["HMXL"], testing=TESTING)
# print(ocn_mxl)
# import pdb; pdb.set_trace()
# datatransformer = DataTransformer(save_path='/glade/work/zespinosa/Projects/SI-Antarctic/data/')
# transform_cesm2_data(
#     datatransformer=datatransformer,
#     ds=ocn_mxl,
#     save=SAVE,
#     save_name="cesm2_ocn-mxl_monthly_1950-01_2023-12",
# )

if __name__ == "__main__":
    transform_cesm2_ens()

    # PRECC, PRECL, QFLX
