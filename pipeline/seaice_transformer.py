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
    def __init__(self, save_path: str = "/glade/work/zespinosa/Projects/SI-Antarctic/data"):
        self.save_path = save_path

    
    def calc_regions(
        self,
        siconc,
        grid,
        regional_bounds=None,
        sanity_check=True,
        polar=False,
        save=False,
        prod="NSIDC",
    ):
        if polar:
            lon, _ = polar_xy_to_lonlat(
                x=siconc.lon,
                y=siconc.lat,
                true_scale_lat=TRUE_SCALE_LATITUDE, 
                re=EARTH_RADIUS_KM, 
                e=EARTH_ECCENTRICITY, 
                hemisphere=SOUTH
            )
        else: 
            lon = siconc.lon
        
        if regional_bounds == None:
            regional_bounds = {
                "ross": (155, 215),
                "amundsen": (215, 255),
                "bellingshausen": (255, 295),
                "weddell": (295, 360),
                "south_indian": (0, 75),
                "south_west_pacific": (75, 155),
            }
        
        si_regions, si_regions_anoms = [], []
        for _, (_, (lonMin, lonMax)) in enumerate(regional_bounds.items()):
            # Mask everything outside of the region
            siconc_region = siconc.where(((lon >= lonMin) & (lon < lonMax)), np.nan)
            # Calculate sia and sie for the region
            si_cur_region, si_cur_region_anoms = self.calc_sia_sie(siconc_region, grid, hem="SH", prod="NSIDC", save=False)
            si_regions.append(si_cur_region)
            si_regions_anoms.append(si_cur_region_anoms)

        # Merge all regions into one dataset for sia and sie
        si_regions = xr.concat(si_regions, dim="region")
        si_regions["region"] = list(regional_bounds.keys())
        # Merge all regions into one dataset for sia and sie anomalies 
        si_regions_anoms = xr.concat(si_regions_anoms, dim="region")
        si_regions_anoms["region"] = list(regional_bounds.keys())

        # Save to netcdf
        if save:
            si_regions.to_netcdf(f"{self.save_path}/si_regions_{prod}.nc")
            si_regions_anoms.to_netcdf(f"{self.save_path}/si_regions_{prod}_anoms.nc")

        return si_regions, si_regions_anoms


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
        prod: str = "CESM",
        save: bool = False,
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

        # NSIDC Native Grid
        if area is None and prod == "NSIDC":
            print("NSIDC Native Grid")
            area = xr.open_dataset("/glade/work/zespinosa/GRIDS/areacello_Bootstrap_polar_stereo_25km_SH.nc")["areacello"]
            area = area.rename({"ygrid": "y", "xgrid": "x"})
            lat, lon = "y", "x"
            div = 1e6
        # CESM Native Grid
        elif area is None and prod == "CESM":
            print("CESM2 Native Grid")
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
        # NSIDC or CESM Regridded
        elif area is not None:
            print("CESM2 or NSIDC Regrid")
            if hem == "SH":
                siconc = siconc.sel(lat=slice(-90, 0))
                area = area.sel(lat=slice(-90, 0))
            else:
                siconc = siconc.sel(lat=slice(0, 90))
                area = area.sel(lat=slice(0, 90))
            lat, lon = "lat", "lon"
            div = 1e6
        else:
            raise ValueError("something is wrong with area")

        # Calculate sia and sie
        sia = ((siconc * area).sum([lat, lon]) / div).rename("sia")
        sie = (xr.where(siconc >= 0.15, area, 0).sum([lat, lon]) / div).rename("sie")
        si = xr.merge([sia.to_dataset(), sie.to_dataset()])

        # Calculate sia and sie anomalies
        ref_period = ("1980-01-01", "2020-01-01")
        freq = "month"
        si = si.bounds.add_bounds("T")
        sia_anoms = si.temporal.departures("sia", freq=freq, reference_period=ref_period)["sia"]
        sie_anoms = si.temporal.departures("sie", freq=freq, reference_period=ref_period)["sie"]
        si_anoms = xr.merge([sia_anoms.to_dataset(), sie_anoms.to_dataset()])

        # Save to netcdf
        if save:
            si.to_netcdf(f"{self.save_path}/si_{prod}_{hem}_si.nc")
            si_anoms.to_netcdf(f"{self.save_path}/si_{prod}_{hem}_si-anoms.nc")
        
        return si, si_anoms


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
        "/glade/derecho/scratch/zespinosa/archive/cesm2.1.3_BHISTcmip6_f09_g17_ERA5_nudge/", 
        "/glade/derecho/scratch/zespinosa/archive/cesm2.1.3_BSSP370cmip6_f09_g17_ERA5_nudge/"
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

def test_si_regrid_nsidc():
    ice_nsidc = dataloader.get_nsidc_data(hem="south")
    ice_nsidc = datatransformer.regrid_polarsterographic(ds=ice_nsidc, hem="south", save=True, save_name="nsidc_regrid")
    # Get anomalies
    # REF_PERIOD = ("1980-01-01", "2020-01-01")
    # ice_nsidc_anoms = datatransformer.calculate_anoms_climatology(
    #     ds=ice_nsidc,
    #     ref_period=REF_PERIOD,
    #     save_name="nsidc_regrid",
    #     save=True,
    # )
    areacello = datatransformer.get_grid_cell_area(ice_nsidc)
    si_nsidc_regions, si_nsidc_regions_anoms = cice_transformer.calc_regions(ice_nsidc["cdr_seaice_conc"], areacello, prod="NSIDC", polar=False, save=True)
    # si_nsidc, si_nsidc_anoms = cice_transformer.calc_sia_sie(ice_nsidc["cdr_seaice_conc"], area=areacello, hem="SH", prod="NSIDC", save=True)
    return si_nsidc_regions, si_nsidc_regions_anoms

def test_si_regrid_cesm2():
    # Verify sia and sie work with CESM2 on Native Grid
    ice_cesm2 = dataloader.get_cesm2_data(
        comp="ice",
        myvars=["aice", "daidtt", "daidtd", "dvidtt", "dvidtd", "sithick", "uvel", "vvel"],
        testing=False,
    )
    ice_cesm2 = datatransformer.regrid(ice_cesm2)
    areacello = datatransformer.get_grid_cell_area(ice_cesm2)
    # Regions
    si_cesm2_regions, si_cesm2_regions_anoms = cice_transformer.calc_regions(ice_cesm2["aice"], areacello, prod="CESM", polar=False, save=True)
    # Raw CESM2
    si_cesm2, si_cesm2_anoms = cice_transformer.calc_sia_sie(ice_cesm2["aice"], area=areacello, hem="SH", prod="CESM", save=True)
    
    return si_cesm2_regions, si_cesm2_regions_anoms

##### Test CESM2 #####
# Native Grid
# si_cesm2 = test_si_native_cesm2()
# print(si_cesm2.sie.values)
# Regrid
# test_si_regrid_nsidc()
test_si_regrid_cesm2()
# si_cems2, si_cesm2_anoms = test_si_regrid_cesm2()
# print(si_cesm2.sie.values)
# 

##### Test NSIDC #####
# test_si_regrid_nsidc()
# si_nsidc = test_si_native_nsidc()
# print(si_nsidc.sie.values[:10])
# si_nsidc = test_si_regrid_nsidc()
# print(si_nsidc.sie.values[:10])

# TEST CESM
#   - Regional SIA and SIE
# si_cesm2 = cice_transformer.(ice_cesm2["aice"], hem="SH", prod="CESM")

# ice_cesm2 = dataloader.regrid(ice_cesm2)
# ice_cesm2_trend = dataloader.calculate_linear_time_trend(ice_cesm2, myvars=["aice"])
# ice_cesm2_ac = dataloader.calculate_anoms_climatology(ice_cesm2, ref_period=("1950-01-01", "1950-02-01"))
