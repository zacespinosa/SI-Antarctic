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


def era5_pressure_level(dataloader):
    myvars = [
        'geopotential',
        'temperature',
        'u_component_of_wind',
        'v_component_of_wind',
    ]
    all_vars = []
    for cvar in myvars:
        era5_data = dataloader.get_era5_data(
            level="pressure",
            info={
                "vars": [cvar],
                "years": [str(yr) for yr in list(range(1979, 2024))],
                "months": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10","11", "12"],
                "area":[90, -180, -90, 180],
                "time":"00:00",
                "pressure_levels": ["1000", "500"],
                "save_name": f"ERA5_monthly_1979-01_2023-12_plevels_{cvar}"
            }
        )
        all_vars.append(era5_data)
    return xr.merge(all_vars)


def era5_single_level(dataloader):
    myvars = [
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        '10m_wind_speed',
        '2m_temperature',
        'mean_sea_level_pressure',
        'sea_surface_temperature',
    ]
    all_vars = []
    for cvar in myvars:
        era5_data = dataloader.get_era5_data(
            level="single",
            info={
                "vars": [cvar],
                "years": [str(yr) for yr in list(range(1979, 2024))],
                "months": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10","11", "12"],
                "area":[90, -180, -90, 180],
                "time":"00:00",
                "save_name": f"ERA5_monthly_1979-01_2023-12_{cvar}"
            }
        )
        all_vars.append(era5_data)

    return xr.merge(all_vars)


#dataloader = DataLoader(
#    root = [
#        "/glade/campaign/univ/uwas0118/scratch/archive/1950_2015/",
#        "/glade/scratch/zespinosa/archive/cesm2.1.3_BHISTcmip6_f09_g17_ERA5_nudge/",
#        "/glade/scratch/zespinosa/archive/cesm2.1.3_BSSP370cmip6_f09_g17_ERA5_nudge/"
#    ],
#    era5_root="/glade/work/zespinosa/data/era5/monthly"
#)

# ERA5
#era5_single_level(dataloader)
#era5_pressure_level(dataloader)

