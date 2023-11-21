"""
Define a class for manipulating CESM2 and ERA5 data

Author: Zac Espinosa
Date: Nov 9, 2023
"""
import os

from glob import glob
from typing import List, Tuple

import numpy as np
import xarray as xr
import xcdat as xc
import xskillscore as xscore

# Personal Data Loader
from data_loader import DataLoader

class DataTransformer():
    """
    This defines a class DataTransformer.
    Capabilities include regridding, calculating anomalies, calculating climatologies, and calculating trends
    """
    def __init__(self, save_path: str = ".") -> None:
        """
        Arguments:
        ----------
        save_path (str): path to save data to
        """
        self.save_path = save_path
        self.skip_vars = ["expver", "latitude", "longitude", "lev", "level", "time", "lon", "lat", "lat_bnds", "lon_bnds", "time_bnds", "lev_bnds"]

    def get_grid_cell_area(
        self,
        ds: xr.Dataset,
    ) -> xr.Dataset:
        # Use xcdat to add lat/lon bounds
        ds = ds.bounds.add_bounds("X")
        ds = ds.bounds.add_bounds("Y")
        # Use bounds to calculate grid cell area
        lat_bnds = ds.lat_bnds
        lon_bnds = ds.lon_bnds

        # Radius of earth in km
        R = 6371 # Radius of earth in km

        # A = pi*R^2*(sin(lat2)-sin(lat1))*(lon2-lon1)/180 where longitudes are in degrees and latitudes are in radians
        areacello = np.pi*(R**2)*(np.sin(np.deg2rad(lat_bnds[:, 1])) - np.sin(np.deg2rad(lat_bnds[:, 0]))) * (lon_bnds[:, 1] - lon_bnds[:, 0])/180

        return areacello



    def regrid(
        self,
        ds: xr.Dataset = False,
        myvars: List[str] = False,
        grid: np.ndarray = False,
        save_name: str = "",
        save: bool = False,
    ) -> xr.Dataset:
        """
        Regrid each variable in ds[myvars] to a new grid

        Arguments:
        ----------
        ds (xr.Dataset): xarray dataset containing myvars
        myvars (List[str]): list of variables to regrid. Defaults to regridding all variables in dataset
        grid (np.ndarray): new grid to regrid to. Defaults to 1x1degree grid

        Returns:
        ----------
        ds (xr.Dataset): xarray dataset containing regridded variables
        """
        save_name = os.path.join(self.save_path, save_name)
        if not ds:
            return xr.open_dataset(f"{save_name}.nc")

        # If myvars is None, regrid all variables in dataset
        if not myvars:
            myvars = [v for v in list(ds.variables.keys()) if v not in self.skip_vars]

        # Use default grid
        if not grid:
            lat = np.arange(-89.5, 90.5, 1)
            lon = np.arange(.5, 360.5, 1)
            grid = xc.create_grid(lat, lon)

        ds_regrid = []
        for cvar in myvars:
            ds_regrid.append(ds.regridder.horizontal(cvar, grid, tool='xesmf', method='bilinear'))

        ds_regrid = xr.merge(ds_regrid)

        if save:
            ds_regrid.to_netcdf(f"{save_name}-regrid.nc")

        return ds_regrid


    def add_coordinate_bounds(self, ds) -> xr.Dataset:
        """
        Add bounds to xarray Dataset. This is necessary for xcdat operations
        """
        ds = ds.bounds.add_bounds("T")
        ds = ds.bounds.add_bounds("X")
        ds = ds.bounds.add_bounds("Y")
        return ds


    def calculate_anoms_climatology(
        self,
        ds: xr.Dataset = False,
        myvars: List[str] = False,
        ref_period: Tuple[str, str] = ("2000-01-01", "2020-01-01"),
        freq: str ="month",
        save_name: str = "",
        save: bool = False,
    ) -> xr.Dataset:
        """
        Calculate anomalies for each variable in ds[myvars]

        Arguments:
        ----------
        ds (xr.Dataset): xarray dataset containing myvars
        myvars (List[str]): list of variables to calculate anomalies for. Defaults to calculating anomalies for all variables in dataset
        ref_period (Tuple[str, str]): reference period to calculate anomalies with respect to (i.e. climatology)
        freq (str): frequency of anomalies to calculate (i.e. month, year, etc.)

        Returns:
        ----------
        ds (xr.Dataset): xarray dataset containing anomalies
        """
        save_name = os.path.join(self.save_path, save_name)
        if not ds:
            ds_anoms = xr.open_dataset(f"{save_name}-clim.nc")
            ds_clim = xr.open_dataset(f"{save_name}-anoms.nc")
            return {"anoms": ds_anoms, "clim": ds_clim}

        # Add time bounds
        ds = ds.bounds.add_bounds("T")

        if not myvars:
            myvars = [v for v in list(ds.variables.keys()) if v not in self.skip_vars]

        ds_clim, ds_anoms = xr.Dataset(), xr.Dataset()
        for cvar in myvars:
            # Calculate anomalies
            ds_anoms[cvar] = ds.temporal.departures(cvar, freq=freq, reference_period=ref_period)[cvar]
            # Calculate Climatology
            ds_clim[cvar] = ds.temporal.climatology(cvar, freq=freq, reference_period=ref_period)[cvar]

        ds_anoms = ds_anoms.assign_attrs(ds.attrs)
        ds_clim = ds_clim.assign_attrs(ds.attrs)

        if save:
            ds_clim.to_netcdf(f"{save_name}-clim.nc")
            ds_anoms.to_netcdf(f"{save_name}-anoms.nc")

        return {"anoms": ds_anoms, "clim": ds_clim}


    def calculate_linear_time_trend(
        self,
        ds: xr.Dataset = False,
        myvars: List[str] = False,
        save_name: str = "",
        save: bool = False,
    ) -> xr.Dataset:
        """
        Calculate anomalies for each variable in ds[myvars]

        Arguments:
        ----------
        ds (xr.Dataset): xarray dataset containing myvars
        myvars (List[str]): list of variables to calculate anomalies for. Defaults to calculating anomalies for all variables in dataset

        Returns:
        ----------
        ds (xr.Dataset): xarray dataset containing anomalies
        """
        save_name = os.path.join(self.save_path, save_name)
        if not ds:
            return xr.open_dataset(f"{save_name}-trends.nc")

        if not myvars:
            myvars = [v for v in list(ds.variables.keys()) if v not in self.skip_vars]

        # rechunk ds along time
        ds = ds.chunk({"time": -1})
        time = xr.DataArray(np.arange(ds.time.shape[0]), dims=["time"])

        ds_trends = xr.Dataset()
        for cvar in myvars:
            ds_trends[cvar] = xscore.linslope(time, ds[cvar], dim="time", skipna=False, keep_attrs=True)

        ds_trends = ds_trends.assign_attrs(ds.attrs)

        if save:
            ds_trends.to_netcdf(f"{save_name}-trends.nc")

        return ds_trends

############################################################################################################
# dataloader = DataLoader(
#     root = [
#         "/glade/campaign/univ/uwas0118/scratch/archive/1950_2015/",
#         "/glade/scratch/zespinosa/archive/cesm2.1.3_BHISTcmip6_f09_g17_ERA5_nudge/",
#         "/glade/scratch/zespinosa/archive/cesm2.1.3_BSSP370cmip6_f09_g17_ERA5_nudge/"
#     ])


####### TESTING #######
# def _test_cesm2_ice(dataloader):
#     ice_cesm2 = dataloader.regrid(ice_cesm2)
#     ice_cesm2_trend = dataloader.calculate_linear_time_trend(ice_cesm2, myvars=["aice"])
#     ice_cesm2_ac = dataloader.calculate_anoms_climatology(ice_cesm2, ref_period=("1950-01-01", "1950-02-01"))
#     print(ice_cesm2_ac["anoms"])

# def _test_cesm2_ocn(dataloader):
#     ocn_mxl = dataloader.regrid(ocn_mxl)
#     ocn_mxl_ac = dataloader.calculate_anoms_climatology(ocn_mxl, ref_period=("1950-01-01", "1950-02-01"))
#     print(ocn_mxl_ac["anoms"])

#     ocn_sst = dataloader.regrid(ocn_sst)
#     ocn_sst_ac = dataloader.calculate_anoms_climatology(ocn_sst, ref_period=("1950-01-01", "1950-02-01"))
#     print(ocn_sst_ac["anoms"])

# def _test_cesm2_atm(dataloader):
#     atm_cesm2 = dataloader.regrid(atm_cesm2)
#     atm_cesm2_ac = dataloader.calculate_anoms_climatology(atm_cesm2, ref_period=("1950-01-01", "1950-02-01"))
#     atm_cesm2_trends = dataloader.calculate_linear_time_trend(atm_cesm2, myvars=["U10", "PSL"])
#     print(atm_cesm2_ac["anoms"])


# def _test_era5_single_level(dataloader, datatransformer):
#     cvar = "10m_u_component_of_wind"
#     era5_data = dataloader.get_era5_data(
#         level="single",
#         info={
#             "vars": [cvar],
#             "years": [str(yr) for yr in list(range(1979, 2023))],
#             "months": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10","11", "12"],
#             "area":[90, -180, -90, 180],
#             "time":"00:00",
#             "save_name": f"ERA5_monthly_1979-01_2023-12_{cvar}"
#         }
#     )
#     era5_data = era5_data.sel(time=slice("1979-01-01", "1980-01-01"))
#     era5_data = datatransformer.regrid(ds=era5_data, myvars=["u10"], save=False)
#     era5_data_ac = datatransformer.calculate_anoms_climatology(
#         ds=False,
#         ref_period=("1979-01-01", "1980-01-01"),
#         save_name=f"ERA5_monthly_1979-01_2023-12_{cvar}",
#         save=False,
#     )
#     print(era5_data_ac["anoms"])
#     era5_data_trends = datatransformer.calculate_linear_time_trend(
#         era5_data,
#         save=True,
#         save_name=f"ERA5_monthly_1979-01_2023-12_{cvar}",
#     )
#     # print(era5_data_trends)

# def test_runner():
#     dataloader = DataLoader(
#         root = [
#             "/glade/campaign/univ/uwas0118/scratch/archive/1950_2015/",
#             "/glade/scratch/zespinosa/archive/cesm2.1.3_BHISTcmip6_f09_g17_ERA5_nudge/",
#             "/glade/scratch/zespinosa/archive/cesm2.1.3_BSSP370cmip6_f09_g17_ERA5_nudge/"
#         ],
#         era5_root="/glade/work/zespinosa/data/era5/monthly"
#     )
#     datatransformer = DataTransformer(
#         save_path='/glade/work/zespinosa/Projects/SI-Antarctic/data')

    # Test ERA5
    # _test_era5_single_level(dataloader, datatransformer)
    # _test_era5_pressure_level(dataloader, datatransformer)

    # Test CESM2
    # _test_cesm2_ice(dataloader)
    # _test_cesm2_ocn(dataloader)
    # _test_cesm2_atm(dataloader)

# test_runner()
