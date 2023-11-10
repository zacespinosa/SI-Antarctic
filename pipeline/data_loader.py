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


class DataLoader():
    """
    This defines a class DataLoader that can be used to load any variable from CESM2 and ERA5
    """
    def __init__(
        self, 
        root: List[str] = [""], 
        era5_root: List[str] = [""], 
    ) -> None:
        """
        Arguments:
        ----------
        root (str or List[str]): path or list of paths to the root directory of the CESM2 data
        era5_root (str or List[str]): path or list of paths to the root directory of the era5 data
        """
        self.root = root
        self.era5_root = era5_root


    def get_era5_data(
        self,
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
        level (str): either "single" or "pressure" to download single level or pressure level data
        info (Dict): dictionary containing information about the data to download

        Returns:
        ----------
        ds (xr.Dataset): dataset of ERA5 data containing variables in myvars
        """
        fp = os.path.join(self.era5_root, info["save_name"])

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
            f'{os.path.join(self.era5_root, info["save_name"])}.nc')
    

    def get_cesm2_data(self, comp: str = "cice", myvars: List = None, levels: List = None, testing: bool = False) -> xr.Dataset:
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
        if testing: files = files[:5]

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




####### TESTING #######
def _test_cesm2_ice(dataloader):
    ice_cesm2 = dataloader.get_cesm2_data(
        comp="ice",
        myvars=["aice", "daidtt", "daidtd", "dvidtt", "dvidtd", "sithick", "uvel", "vvel"],
        testing=True
    )
    print(ice_cesm2)

def _test_era5_single_level(dataloader):
    myvars = [
        '10m_u_component_of_wind', 
        '10m_v_component_of_wind',
        '10m_wind_speed',
        '2m_temperature', 
        'mean_sea_level_pressure', 
        'sea_surface_temperature',
    ]
    for cvar in myvars:
        era5_data = dataloader.get_era5_data(
            level="single", 
            info={
                "vars": [cvar],
                "years": [str(yr) for yr in list(range(1979, 2023))],
                "months": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10","11", "12"],
                "area":[90, -180, -90, 180],
                "time":"00:00",
                "save_name": f"ERA5_monthly_1979-01_2023-12_{cvar}"
            }
        )
        print(era5_data)
        break

def _test_era5_pressure_level(dataloader):
    myvars = [
        'geopotential', 
        'temperature', 
        'u_component_of_wind',
        'v_component_of_wind',
    ]
    for cvar in myvars:
        era5_data = dataloader.get_era5_data(
            level="pressure", 
            info={
                "vars": [cvar],
                "years": [str(yr) for yr in list(range(1979, 2023))],
                "months": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10","11", "12"],
                "area":[90, -180, -90, 180],
                "time":"00:00",
                "pressure_levels": ["1000", "500"],
                "save_name": f"ERA5_monthly_1979-01_2023-12_plevels_{cvar}"
            }
        )
        print(era5_data)
        break

def _test_cesm2_ocn(dataloader):
    ocn_sst = dataloader.get_data(comp="ocn", myvars=["SST"], testing=True)
    print(ocn_sst)
    ocn_mxl = dataloader.get_data(comp="ocn", myvars=["HMXL"], testing=True)
    print(ocn_mxl)

def _test_cesm2_atm(dataloader):
    atm_cesm2 = dataloader.get_cesm2_data(
        comp="atm", 
        myvars=["PSL", "U10", "TS", "T", "U", "V", "Z3"], 
        levels=[1000, 850, 500],
        testing=True
    )
    print(atm_cesm2)


def test_runner():
    dataloader = DataLoader(
        root = [
            "/glade/campaign/univ/uwas0118/scratch/archive/1950_2015/",
            "/glade/scratch/zespinosa/archive/cesm2.1.3_BHISTcmip6_f09_g17_ERA5_nudge/", 
            "/glade/scratch/zespinosa/archive/cesm2.1.3_BSSP370cmip6_f09_g17_ERA5_nudge/"
        ],
        era5_root="/glade/work/zespinosa/data/era5/monthly"
    )

    # Test ERA5
    _test_era5_single_level(dataloader)
    _test_era5_pressure_level(dataloader)

    # Test CESM2
    _test_cesm2_ice(dataloader)
    _test_cesm2_ocn(dataloader)
    _test_cesm2_atm(dataloader)

# test_runner()
    

