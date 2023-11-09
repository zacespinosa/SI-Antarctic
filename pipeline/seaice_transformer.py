"""
Define a class for manipulating sea ice data from CESM2 and ERA5 

Author: Zac Espinosa 
Date: Nov 9, 2023
"""
import os
from typing import List, Tuple, Dict

import numpy as np
import xarray as xr
import xcdat as xc


class SeaIceTransformer():
    """
    This class contains utility functions for sea ice data
    """
    def __init__(self):
        pass


    def calc_sia_sie(
        self,
        area: xr.DataArray,
        siconc: xr.DataArray,
        hem: str = "NH",
        prod: str = "CESM"
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Calc SIE and SIA for CESM2 and NSIDC data.
        Arguments:
        ----------
        Returns:
        ----------
        sie (xr.Dataset)
        """
        if prod == "CESM":
            lat_mid_index = int(len(siconc.lat)/2)
            if hem == "NH":
                siconc = siconc[:, lat_mid_index:, :]
                area = area[lat_mid_index:, :]
            if hem == "SH":
                siconc = siconc[:, :lat_mid_index, :]
                area = area[:lat_mid_index, :]
            lat, lon = "nlat", "nlon"
            div = 1e12
        else:
            lat, lon = "latitude", "longitude"
            div = 1e6

        # Calculate sia and sie
        sia = ((siconc * area).sum([lat, lon]) / div).rename("sia")
        sie = (xr.where(siconc >= 0.15, area, 0).sum([lat, lon]) / div).rename("sie")
        
        return sia.to_dataset(), sie.to_dataset()


    def find_ice_edge(self) -> xr.Dataset:
        """
        Arguments:
        ----------
        Returns:
        ----------
        """
        pass


dataloader = DataLoader(
    root = [
        "/glade/campaign/univ/uwas0118/scratch/archive/1950_2015/",
        "/glade/scratch/zespinosa/archive/cesm2.1.3_BHISTcmip6_f09_g17_ERA5_nudge/", 
        "/glade/scratch/zespinosa/archive/cesm2.1.3_BSSP370cmip6_f09_g17_ERA5_nudge/"
    ])

####### TESTING #######





# ice_cesm2 = dataloader.get_data(
#     comp="ice",
#     myvars=["aice", "daidtt", "daidtd", "dvidtt", "dvidtd", "sithick", "uvel", "vvel"]
# )
# ice_cesm2 = dataloader.regrid(ice_cesm2)
# ice_cesm2_trend = dataloader.calculate_linear_time_trend(ice_cesm2, myvars=["aice"])
# ice_cesm2_ac = dataloader.calculate_anoms_climatology(ice_cesm2, ref_period=("1950-01-01", "1950-02-01"))

# ocn_sst = dataloader.get_data(comp="ocn", myvars=["SST"])
# ocn_sst = dataloader.regrid(ocn_sst)
# ocn_sst_ac = dataloader.calculate_anoms_climatology(ocn_sst, ref_period=("1950-01-01", "1950-02-01"))
# print(ocn_sst_ac["anoms"])

# ocn_mxl = dataloader.get_data(comp="ocn", myvars=["HMXL"])
# ocn_mxl = dataloader.regrid(ocn_mxl)
# ocn_mxl_ac = dataloader.calculate_anoms_climatology(ocn_mxl, ref_period=("1950-01-01", "1950-02-01"))
# print(ocn_mxl_ac["anoms"])

# atm_cesm2 = dataloader.get_cesm2_data(comp="atm", myvars=["PSL", "U10", "TS", "T", "U", "V", "Z3"], levels=[1000, 850, 500])
# atm_cesm2 = dataloader.regrid(atm_cesm2)
# atm_cesm2_ac = dataloader.calculate_anoms_climatology(atm_cesm2, ref_period=("1950-01-01", "1950-02-01"))
# atm_cesm2_trends = dataloader.calculate_linear_time_trend(atm_cesm2, myvars=["U10", "PSL"])
# print(atm_cesm2_ac["anoms"])

# atm => h0.*
# cice => h.*
# pop => h.nday1.* = SST or h. = HMXL
    