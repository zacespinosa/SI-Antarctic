"""
Define a class for loading CESM2 and ERA5 data

Author: Zac Espinosa 
Date: Nov 9, 2023
"""
import os

from glob import glob
from typing import List, Tuple, Dict

import cdsapi
import pandas as pd
import numpy as np
import xarray as xr
import xcdat as xc
import xskillscore as xscore


class DataLoader():
    """
    This defines a class DataLoader that can be used to load any variable from CESM2 and ERA5
    """
    def __init__(
        self, 
        root: List[str] = [""], 
        era5_root: List[str] = [""], save_path: str = "."
    ) -> None:
        """
        Arguments:
        ----------
        root (str or List[str]): path or list of paths to the root directory of the CESM2 data
        """
        self.root = root
        self.save_path = save_path


    def get_era5_data(
        self,
        myvars: List[str] = None,
        level: str = "single",
        info: Dict = {},
    ) -> xr.Dataset:
        """
        Either: 
            - 1) Load ERA5 data variable(s) from era5_root or 
            - 2) if it doesn't exist, call CDS API to download the data and save it to era5_root
        Finally return ERA5 data

        Arguments:
        ----------
        myvars (List[str]): list of variables to download
        level (str): either "single" or "pressure" to download single level or pressure level data
        info (Dict): dictionary containing information about the data to download

        Returns:
        ----------
        ds (xr.Dataset): dataset of ERA5 data containing variables in myvars
        """
        fp = os.path.join(self.save_path, info["save_name"])

        # Check if file exists
        if os.path.exists(f"{fp}.nc"):
            return xr.open_dataset(f"{fp}.nc")
        else:
            # Download the data from CDS API, save and return output
            ds = self._call_cdsapi(level=level, info=info)
            return ds


    def _call_cdsapi(self, level: str = "single", info: Dict = {}) -> xr.Dataset:
        """
        Class CDS API to download ERA5 data

        Arguments:
        ----------
        level (str): either "single" or "pressure" to download single level or pressure level data
        info (Dict): dictionary containing information about the data to download

        """
        client = cdsapi.Client()

        # Set defaults
        keys = info.keys()
        assert "vars" in keys, "vars must be a list containing one or more variable name (e.g. ['2m_temperature'])"

        if level == "single":
            info["level"] = "reanalysis-era5-single-levels-monthly-means"
        if level == "pressure":
            info["level"] = "reanalysis-era5-pressure-levels-monthly-means"
            if "pressure_levels" not in keys:
                info["pressure_levels"] = ['1000']


        if "years" not in keys: info["years"] = [str(yr) for yr in list(range(1979, 2022))]
        if "month" not in keys: info["months"] = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10","11", "12"]
        if "area" not in keys: info["area"] = [90, -180, -90, 180]
        if "time" not in keys: info["time"] = "00:00"

        api_call_args = {
                'product_type': 'monthly_averaged_reanalysis',
                'variable': info["vars"],
                'year': info["years"],
                'month': info["months"] ,
                'time': info["time"],
                'area': info["area"],
                'format': 'netcdf',
        }
        if level == "pressure":
            api_call_args["pressure_level"] = info["pressure_levels"]

        return client.retrieve(
            info["level"],
            api_call_args,
            f'{os.path.join(self.save_path, info["save_name"])}.nc')
    

    def get_cesm2_data(self, comp: str = "cice", myvars: List = None, levels: List = None) -> xr.Dataset:
        """
        Load data from any component in CESM2. Lazy load so that data is not immediately stored in memory. Then compute later
        
        Arguments:
        ----------
        comp (str): cesm component to download data from (e.g. ice, atm, ocn, clp, lnd, rof, wav)
        myvars (str or List[str]): path or list of paths to the root directory of the CESM2 data
        drop_date (str): date of file to load datas for dropping. This is useful for dropping variables that we are not interested in analyzing

        Returns:
        ----------
        cesm2 (xr.Dataset): xarray dataset containing all variables from the specified component

        """
        assert myvars != None, "myvars must be a list containing one or more variable name (e.g. ['aice', 'sithick'])"

        if comp == "ice" or (comp == "ocn" and "SST" not in myvars): 
            h = "h"
        elif comp == "atm":
            h = "h0"
        elif comp == "ocn" and "SST" in myvars:
            h = "h.nday1"
        else:
            raise ValueError("comp must be one of cice, atm, or ocn")

        files = []
        for r in self.root:
            cfiles = sorted(glob(os.path.join(r, comp, "hist", f"*.{h}.*.nc")))
            files += cfiles
        files = sorted(files)

        # Remove files that contain SST data from the ocean component and we're not interested in SST databbb
        if comp == "ocn" and h == "h": 
            files = [f for f in files if ("nday1" not in f) and ("once" not in f)]

        print(f"Loading {comp} data from {len(files)} files...")
        files = files[:2]

        # Lazy load data
        cesm2 = xr.open_mfdataset(files, coords="minimal", parallel=True) 

        # Rename lat and lon coordinates (annoyingly, this is different for each component)
        if comp == "ice":
            cesm2 = cesm2.rename({"TLON": "lon", "TLAT": "lat", "nj": "nlat", "ni": "nlon"})
        if comp == "ocn":
            cesm2 = cesm2.rename({"TLONG": "lon", "TLAT": "lat"}) 

        if comp == "ocn" or comp == "ice":
            myvars = [*myvars, "nlat", "nlon"]
        if comp == "atm":
            myvars = ["lev", *myvars]

        # Drop all unwanted variables
        drop_vars = [cvar for cvar in list(cesm2.variables.keys()) if cvar not in ["time", "lon", "lat" ,*myvars]]
        cesm2 = cesm2.drop_vars(drop_vars)

        cesm2["time"] = [pd.to_datetime(f"{t.year}-{t.month}-15") - pd.DateOffset(months=1) for t in cesm2.time.values]
        cesm2['time'].encoding['calendar'] = 'standard'

        # Select levels for atm component
        if levels != None:
            cesm2 = cesm2.sel(lev=levels, method="nearest")

        return cesm2


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
        

    def regrid(
        self,
        ds: xr.Dataset, 
        myvars: List[str] = None, 
        grid: np.ndarray = None,
        save_name: str = "", 
        save: bool = False,
        load: bool = False,
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
        if load: 
            return xr.open_dataset(f"{save_name}.nc")
            
        # If myvars is None, regrid all variables in dataset
        if myvars == None:
            myvars = [v for v in list(ds.variables.keys()) if v not in ["lev", "time", "lon", "lat", "lat_bnds", "lon_bnds", "time_bnds"]]

        # Use default grid
        if grid == None:
            lat = np.arange(-89.5, 90.5, 1)
            lon = np.arange(.5, 360.5, 1)
            grid = xc.create_grid(lat, lon)

        ds_regrid = []
        for cvar in myvars:
            ds_regrid.append(ds.regridder.horizontal(cvar, grid, tool='xesmf', method='bilinear'))

        ds_regrid = xr.merge(ds_regrid)

        if save:
            ds_regrid.to_netcdf(f"{save_name}.nc")

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
        ds: xr.Dataset, 
        myvars: List[str] = None, 
        ref_period: Tuple[str, str] = ("2000-01-01", "2020-01-01"),
        freq: str ="month",
        save_name: str = "",
        save: bool = False,
        load: bool = False,
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
        if load: 
            ds_anoms = xr.open_dataset(f"{save_name}-clim.nc")
            ds_clim = xr.open_dataset(f"{save_name}-anoms.nc")
            return {"anoms": ds_anoms, "clim": ds_clim}

        # Add time bounds
        ds = ds.bounds.add_bounds("T")

        if myvars == None: 
            myvars = [v for v in list(ds.variables.keys()) if v not in ["lev", "time", "lon", "lat", "lat_bnds", "lon_bnds", "time_bnds"]]
        
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
        ds: xr.Dataset, 
        myvars: List[str] = None,
        save_name: str = "",
        save: bool = False,
        load: bool = False,
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
        if load: 
            return xr.open_dataset(f"{save_name}-trends.nc")

        if myvars == None:
            myvars = [v for v in list(ds.variables.keys()) if v not in ["lev", "time", "lon", "lat", "lat_bnds", "lon_bnds", "time_bnds"]]

        # rechunk ds along time 
        ds = ds.chunk({"time": -1})

        ds_trends = xr.Dataset()
        for cvar in myvars:
            ds_trends[cvar] = xscore.linslope(ds.time, ds[cvar], dim="time", keep_attrs=True)

        ds_trends = ds_trends.assign_attrs(ds.attrs)

        if save:
            ds_trends.to_netcdf(f"{save_name}-trends.nc")

        return ds_trends


class SeaIceTransformer():
    """
    This class contains 
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
    