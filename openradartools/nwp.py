import os
from glob import glob
from datetime import datetime

import pyart
import cftime
import numpy as np
import xarray as xr
import pandas as pd
from scipy.interpolate import interp1d

def nwp_profile(radar, source='era5'):
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
        
    if source == 'access':
        if request_dt < datetime.strptime('20200924', '%Y%m%d'):
            #APS2
            access_root = '/g/data/lb4/ops_aps2/access-g/1' #access g
            flip = False
        else:
            #APS3
            access_root = '/g/data/wr45/ops_aps3/access-g/1' #access g
            flip = True
        #build folder for access data
        model_timestep_hr = 6
        hour_folder = str(round(request_dt.hour/model_timestep_hr)*model_timestep_hr).zfill(2) + '00'
        if hour_folder == '2400':
            hour_folder = '0000'
        access_folder = '/'.join([access_root, datetime.strftime(request_dt, '%Y%m%d'), hour_folder, 'an', 'pl'])
        #build filenames
        temp_ffn = access_folder + '/air_temp.nc'
        geop_ffn = access_folder + '/geop_ht.nc'
        rh_ffn   = access_folder + '/relhum.nc'
        
        #check if files exist
        if not os.path.isfile(temp_ffn):
            raise FileNotFoundError(f'{temp_ffn}: no such file for temperature.')
        if not os.path.isfile(geop_ffn):
            raise FileNotFoundError(f'{geop_ffn}: no such file for geopotential.')
        if not os.path.isfile(rh_ffn):
            raise FileNotFoundError(f'{rh_ffn}: no such file for RH.')            

        #extract data
        with xr.open_dataset(temp_ffn) as temp_ds:
            temp_profile = temp_ds.air_temp.sel(lon=request_lon, method='nearest').sel(lat=request_lat, method='nearest').data[0] - 273.15 #units: deg C
        with xr.open_dataset(geop_ffn) as geop_ds:
            geopot_profile = geop_ds.geop_ht.sel(lon=request_lon, method='nearest').sel(lat=request_lat, method='nearest').data[0] #units: m
            pres_profile = geop_ds.level.data[:]/100 #units: Pa
        with xr.open_dataset(rh_ffn) as rh_ds:
            rh_profile = rh_ds.relhum.sel(lon=request_lon, method='nearest').sel(lat=request_lat, method='nearest').data[0] #units: percentage       

    elif source == "era5":
        flip = True
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
            pres_profile = geop_ds.level.data[:] #units: hpa
        with xr.open_dataset(rh_ffn) as rh_ds:
            rh_profile = rh_ds.r.sel(longitude=request_lon, method='nearest').sel(latitude=request_lat, method='nearest').sel(time=request_dt, method='nearest').data[:] #units: percentage     
        
    #flipdata (ground is first row)
    if flip:
        temp_profile = np.flipud(temp_profile)
        geopot_profile = np.flipud(geopot_profile)   
        pres_profile = np.flipud(pres_profile)   
        rh_profile = np.flipud(rh_profile)
    
    #append surface data using lowest level
    geopot_profile = np.append([0], geopot_profile)
    pres_profile = np.append(pres_profile[0], pres_profile)
    temp_profile = np.append(temp_profile[0], temp_profile)
    rh_profile = np.append(rh_profile[0], rh_profile)
    
    
    #map temp and z to radar gates
    z_field, temp_field = pyart.retrieve.map_profile_to_gates(temp_profile, geopot_profile, radar)
    temp_info_field = {'data': temp_field['data'],  # Switch to celsius.
                      'long_name': 'Sounding temperature at gate',
                      'standard_name': 'temperature',
                      'valid_min': -100, 'valid_max': 100,
                      'units': 'degrees Celsius',
                      'comment': 'Radiosounding date: %s' % (request_dt.strftime("%Y/%m/%d"))}
    
    #generate isom dataset
    melting_level = find_melting_level(temp_profile, geopot_profile)
    # retrieve the Z coordinates of the radar gates
    rg, azg = np.meshgrid(radar.range['data'], radar.azimuth['data'])
    rg, eleg = np.meshgrid(radar.range['data'], radar.elevation['data'])
    _, _, z = pyart.core.antenna_to_cartesian(rg / 1000.0, azg, eleg)
    #calculate height above melting level
    isom_data = (radar.altitude['data'] + z) - melting_level
    isom_data[isom_data<0] = 0
    isom_field = {'data': isom_data, # relative to melting level
                      'long_name': 'Height relative to (H0+H10)/2 level',
                      'standard_name': 'relative_melting_level_height',
                      'units': 'm'}
    
    #interpolate to 0C and -20C levels
    fz_level = np.round(_sounding_interp(temp_profile, geopot_profile, 0))
    minus_20_level = np.round(_sounding_interp(temp_profile, geopot_profile, -20))
    levels_dict = {'fz_level':fz_level, 'minus_20_level':minus_20_level}
    
    #insert original profiles into a dictionary
    profile_dict = {'t':temp_profile, 'z':geopot_profile, 'r':rh_profile, 'p':pres_profile}
    
    
    return  z_field, temp_info_field, isom_field, profile_dict, levels_dict

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

def find_melting_level(temp_profile, geop_profile):
    #interpolate to required levels
    plus10_h = _sounding_interp(temp_profile, geop_profile, 10.)
    fz_h = _sounding_interp(temp_profile, geop_profile, 0.)
    #calculate base of melting level
    return (plus10_h+fz_h)/2