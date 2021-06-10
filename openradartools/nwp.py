import os
from glob import glob

import cftime
import numpy as np
import xarray as xr
import pandas as pd
from scipy.interpolate import interp1d

def nwp_profile(request_dt, request_lat, request_lon):
    """
    Compute the signal-to-noise ratio as well as interpolating the radiosounding
    temperature on to the radar grid. The function looks for the radiosoundings
    that happened at the closest time from the radar. There is no time
    difference limit.
    Parameters:
    ===========
    radar:
        Py-ART radar object.
    Returns:
    ========
    z_field: dict
        Altitude in m, interpolated at each radar gates.
    temp_info_field: dict
        Temperature in Celsius, interpolated at each radar gates.
    profile_dict: dict
        contains profiles of Z, T and RH
    level_dict: dict
        freezing/-20C levels (2x floats)
    """
    #get radar metadata
    request_lat = radar.latitude['data'][0]
    request_lon = radar.longitude['data'][0]
    request_dt = pd.Timestamp(cftime.num2pydate(radar.time['data'][0], radar.time['units']))
        
    #set era path
    era5_root = '/g/data/rt52/era5/pressure-levels/reanalysis'
    
    #build file paths
    month_str = str(request_dt.month).zfill(2)
    year_str = str(request_dt.year)
    temp_ffn = glob(f'{era5_root}/t/{year_str}/t_era5_oper_pl_{year_str}{month_str}*.nc')[0]
    geop_ffn = glob(f'{era5_root}/z/{year_str}/z_era5_oper_pl_{year_str}{month_str}*.nc')[0]
    rh_ffn   = glob(f'{era5_root}/r/{year_str}/r_era5_oper_pl_{year_str}{month_str}*.nc')[0]
    
    #extract data
    with xr.open_dataset(temp_ffn) as temp_ds:
        temp_profile = temp_ds.t.sel(longitude=request_lon, method='nearest').sel(latitude=request_lat, method='nearest').sel(time=request_dt, method='nearest').data[:] - 273.15 #units: deg K -> C
    with xr.open_dataset(geop_ffn) as geop_ds:
        geopot_profile = geop_ds.z.sel(longitude=request_lon, method='nearest').sel(latitude=request_lat, method='nearest').sel(time=request_dt, method='nearest').data[:]/9.80665 #units: m**2 s**-2 -> m
    with xr.open_dataset(rh_ffn) as rh_ds:
        rh_profile = rh_ds.r.sel(longitude=request_lon, method='nearest').sel(latitude=request_lat, method='nearest').sel(time=request_dt, method='nearest').data[:] #units: percentage        
        
    #flipdata (ground is first row)
    temp_profile = np.flipud(temp_profile)
    geop_data = np.flipud(geopot_profile)
    rh_profile = np.flipud(rh_profile)
    
    #map to radar
    z_field, temp_field = pyart.retrieve.map_profile_to_gates(temp_profile, geopot_profile, radar)
    temp_info_field = {'data': temp_field['data'],  # Switch to celsius.
                      'long_name': 'Sounding temperature at gate',
                      'standard_name': 'temperature',
                      'valid_min': -100, 'valid_max': 100,
                      'units': 'degrees Celsius',
                      'comment': 'Radiosounding date: %s' % (dtime.strftime("%Y/%m/%d"))}
    
    #interpolate to 0C and -20C levels
    fz_level = np.round(_sounding_interp(temp_profile, geopot_profile, 0))
    minus_20_level = np.round(_sounding_interp(temp_profile, geopot_profile, -20))    
    profile_dict = {'t':temp_profile, 'z':geopot_profile, 'r':rh_profile}
    levels_dict = {'fz_level':fz_level, 'minus_20_level':minus_20_level}
    
    return  z_field, temp_info_field, profile_dict, levels_dict

def _sounding_interp(snd_temp, snd_height, target_temp):
    """
    Provides an linear interpolated height for a target temperature using a
    sounding vertical profile. Looks for first instance of temperature
    below target_temp from surface upward.

    Parameters
    ----------
    snd_temp : ndarray
        Temperature data (degrees C).
    snd_height : ndarray
        Relative height data (m).
    target_temp : float
        Target temperature to find height at (m).

    Returns
    -------
    intp_h: float
        Interpolated height of target_temp (m).

    """
    intp_h = np.nan

    #check if target_temp is warmer than lowest level in sounding
    if target_temp>snd_temp[0]:
        print('warning, target temp level below sounding, returning ground level (0m)')
        return 0.
    
    # find index above and below freezing level
    mask = np.where(snd_temp < target_temp)
    above_ind = mask[0][0]

    # index below
    below_ind = above_ind - 1
    
    # apply linear interplation to points above and below target_temp
    set_interp = interp1d(
        snd_temp[below_ind:above_ind+1],
        snd_height[below_ind:above_ind+1], kind='linear')
    
    # apply interpolant
    intp_h = set_interp(target_temp)
    
    return intp_h