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

from data_loader import DataLoader
from data_transformer import DataTransformer


class SeaIceTransformer():
    """
    This class contains utility functions for sea ice data
    """
    def __init__(self):
        pass

    
    def calc_regions(
        self,
        siconc,
        grid, 
        regional_bounds=None, 
        sanity_check=True
    ):
        lon, lat = polar_xy_to_lonlat(
            x=siconc.longitude, 
            y=siconc.latitude, 
            true_scale_lat=TRUE_SCALE_LATITUDE, 
            re=EARTH_RADIUS_KM, 
            e=EARTH_ECCENTRICITY, 
            hemisphere=SOUTH
        )
        
        if regional_bounds == None:
            regional_bounds = {
                "ross": (155, 215),
                "amundsen": (215, 255),
                "bellingshausen": (255, 295),
                "weddell": (295, 360),
                "south_indian": (0, 75),
                "south_west_pacific": (75, 155),
            }
        
        siObs_regions = []
        for i, (reg, (lonMin, lonMax)) in enumerate(regional_bounds.items()):
            siconc_region = siconc.where(((lon >= lonMin) & (lon < lonMax)), np.nan)
            siObs_regions.append(calc_sia_and_sie_nsidc(siconc_region, grid))
            
        siObs_regions = xr.concat(siObs_regions, dim="region")
        siObs_regions["region"] = list(regional_bounds.keys())
        return siObs_regions


    def region_regions(self) -> xr.Dataset:
        """
        Mask data to get regions
        Arguments:
        ----------
        Returns:
        ----------
        siconc regions (xr.Dataset)
        """
        pass

    def calc_sia_sie(
        self,
        siconc: xr.DataArray,
        area: xr.DataArray = None,
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
        assert prod in ["CESM", "NSIDC"], "prod must be either CESM or NSIDC"

        if area is None and prod == "NSIDC":
            area = xr.open_dataset("/glade/work/zespinosa/GRIDS/areacello_Bootstrap_polar_stereo_25km_SH.nc")["areacello"]
            area = area.rename({"ygrid": "y", "xgrid": "x"})
            lat, lon = "y", "x"
            div = 1e6

        if area is None and prod == "CESM":
            area = xr.open_dataset("/glade/work/zespinosa/GRIDS/areacello_Ofx_CESM2_historical_r1i1p1f1_gn.nc")["areacello"]

            lat_mid_index = int(len(siconc.lat)/2)
            if hem == "NH":
                siconc = siconc[:, lat_mid_index:, :]
                area = area[lat_mid_index:, :]
            if hem == "SH":
                siconc = siconc[:, :lat_mid_index, :]
                area = area[:lat_mid_index, :]
            lat, lon = "nlat", "nlon"
            div = 1e12

        if area is not None:
            if hem == "SH":
                siconc = siconc.sel(lat=slice(-90, 0))
                area = area.sel(lat=slice(-90, 0))
            else:
                siconc = siconc.sel(lat=slice(0, 90))
                area = area.sel(lat=slice(0, 90))
            lat, lon = "lat", "lon"
            div = 1e6

        # Calculate sia and sie
        sia = ((siconc * area).sum([lat, lon]) / div).rename("sia")
        sie = (xr.where(siconc >= 0.15, area, 0).sum([lat, lon]) / div).rename("sie")
        
        return xr.merge([sia.to_dataset(), sie.to_dataset()])


    def find_ice_edge(self) -> xr.Dataset:
        """
        Finds the sea ice edge defined by the region closest to 15% sea ice concentration
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

datatransformer = DataTransformer(
    save_path='/glade/work/zespinosa/Projects/SI-Antarctic/data'
)

####### TESTING #######
cice_transformer = SeaIceTransformer()

# Verify sia and sie work with CESM2 on Native Grid
def test_si_native_cesm2():
    ice_cesm2 = dataloader.get_cesm2_data(
        comp="ice",
        myvars=["aice", "daidtt", "daidtd", "dvidtt", "dvidtd", "sithick", "uvel", "vvel"],
        testing=True,
    )
    si_cesm2 = cice_transformer.calc_sia_sie(ice_cesm2["aice"], hem="SH", prod="CESM")
    return si_cesm2

def test_si_native_nsidc():
    # Verify sia and sie work with NSIDC on Native Grid
    ice_nsidc = dataloader.get_nsidc_data(hem="south")
    si_nsidc = cice_transformer.calc_sia_sie(ice_nsidc["cdr_seaice_conc"], hem="SH", prod="NSIDC")
    return si_nsidc

def test_si_regrid_cesm2():
    # Verify sia and sie work with CESM2 on Native Grid
    ice_cesm2 = dataloader.get_cesm2_data(
        comp="ice",
        myvars=["aice", "daidtt", "daidtd", "dvidtt", "dvidtd", "sithick", "uvel", "vvel"],
        testing=True,
    )
    ice_cesm2 = datatransformer.regrid(ice_cesm2)
    areacello = datatransformer.get_grid_cell_area(ice_cesm2)
    si_cesm2 = cice_transformer.calc_sia_sie(ice_cesm2["aice"], area=areacello, hem="SH", prod="CESM")
    return si_cesm2

si_cesm2 = test_si_native_cesm2()
print(si_cesm2.sia.values)
si_cesm2 = test_si_regrid_cesm2()
print(si_cesm2.sia.values)

# TEST CESM
#   - Regional SIA and SIE
# si_cesm2 = cice_transformer.(ice_cesm2["aice"], hem="SH", prod="CESM")

# ice_cesm2 = dataloader.regrid(ice_cesm2)
# ice_cesm2_trend = dataloader.calculate_linear_time_trend(ice_cesm2, myvars=["aice"])
# ice_cesm2_ac = dataloader.calculate_anoms_climatology(ice_cesm2, ref_period=("1950-01-01", "1950-02-01"))
