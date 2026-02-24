import os
from glob import glob
from datetime import datetime

import pyart
import cftime
import numpy as np
import xarray as xr
import pandas as pd
from scipy.interpolate import interp1d

import openradartools as ort

def nwp_temperature_levels(request_dt, radar_id, t_levels=[0, -10, -20]):
    
    """
    Extracts temperature profiles from ERA5 for a given radar site and time, and interpolates to specified temperature levels.
    """
    
    #set era path
    era5_root = '/g/data/rq0/admin/era5_site_profiles'
    #build file paths
    profile_ffn = f'{era5_root}/{radar_id}/{radar_id}_era5_profiles_{request_dt.strftime("%Y%m")}.nc'
    ort.file.check_file_exists(profile_ffn)
    
    #extract data
    with xr.open_dataset(profile_ffn) as ds:
        temp_data = ds.t.sel(time=request_dt, method='nearest').data[:]
        geop_data = ds.z.sel(time=request_dt, method='nearest').data[:]
        
    #flipdata (ground is first row)
    temp_data = np.flipud(temp_data)
    geop_data = np.flipud(geop_data)
    
    #interpolate to levels
    output = []
    for level in t_levels:
        output.append(np.round(ort.nwp.sounding_interp(temp_data, geop_data, level)))
    
    return output


def nwp_profile(radar, source='era5',
                radar_id=None,
                override_dt=None,
                access_root_aps2='/g/data/lb4/ops_aps2/access-g/1',
                access_root_aps3='/g/data/wr45/ops_aps3/access-g/1',
                access_root_aps4='/g/data/wr45/ops_aps3/access-g/1',
                era5_pl_root='/g/data/rt52/era5/pressure-levels/reanalysis',
                era5_sl_root = '/g/data/rt52/era5/single-levels/reanalysis',
                era5_profile_root = '/g/data/rq0/admin/era5_site_profiles',
                ):
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
    request_alt = radar.altitude['data'][0]
    if override_dt is None:
        request_dt = pd.Timestamp(cftime.num2pydate(radar.time['data'][0], radar.time['units']))
    else:
        print('Warning, using a different date for NWP data. Ensure this option is required')
        request_dt = override_dt
    if source == 'access':
        if request_dt < datetime.strptime('20200924', '%Y%m%d'):
            #APS2
            access_root = access_root_aps2 #access g
        else:
            #APS3
            access_root = access_root_aps3 #access g
        #build folder for access data
        model_timestep_hr = 6
        hour_folder = str(round(request_dt.hour/model_timestep_hr)*model_timestep_hr).zfill(2) + '00'
        if hour_folder == '2400':
            hour_folder = '0000'
        access_folder = '/'.join([access_root, datetime.strftime(request_dt, '%Y%m%d'), hour_folder, 'an'])
        
        #build filenames
        temp_pl_ffn = access_folder + '/pl/air_temp.nc'
        geop_pl_ffn = access_folder + '/pl/geop_ht.nc'
        rh_pl_ffn   = access_folder + '/pl/relhum.nc'
        sfc_pres_ffn = access_folder + '/sfc/sfc_pres.nc'
                
        #check if files exist
        ort.file.check_file_exists(temp_pl_ffn)
        ort.file.check_file_exists(geop_pl_ffn)
        ort.file.check_file_exists(rh_pl_ffn)
        ort.file.check_file_exists(sfc_pres_ffn)
            
        #extract data
        with xr.open_dataset(temp_pl_ffn) as temp_ds:
            temp_profile = temp_ds.air_temp.sel(lon=request_lon, method='nearest').sel(lat=request_lat, method='nearest').data[0] - 273.15 #units: deg C
        with xr.open_dataset(geop_pl_ffn) as geop_ds:
            geopot_profile = geop_ds.geop_ht.sel(lon=request_lon, method='nearest').sel(lat=request_lat, method='nearest').data[0] #units: m
            pres_profile = geop_ds.lvl.data[:]/100 #units: Pa
        with xr.open_dataset(rh_pl_ffn) as rh_ds:
            rh_profile = rh_ds.relhum.sel(lon=request_lon, method='nearest').sel(lat=request_lat, method='nearest').data[0] #units: percentage       
        with xr.open_dataset(sfc_pres_ffn) as sfc_pres_ds:
            sfc_pres = sfc_pres_ds.sfc_pres.sel(lon=request_lon, method='nearest').sel(lat=request_lat, method='nearest').data[0]/100 #units: Pa       

        
    elif source == "era5_radarsite":
        if radar_id is None:
            raise Exception('missing radar_id value for era5_profiles')
        #build file paths
        profile_ffn = f'{era5_profile_root}/{radar_id}/{radar_id}_era5_profiles_{request_dt.strftime("%Y%m")}.nc'
        ort.file.check_file_exists(profile_ffn)
        #load data
        with xr.open_dataset(profile_ffn) as ds:
            temp_profile = ds.t.sel(time=request_dt, method='nearest').data[:]
            geopot_profile = ds.z.sel(time=request_dt, method='nearest').data[:]
            pres_profile = ds.level.data[:] #hPa
            rh_profile = ds.r.sel(time=request_dt, method='nearest').data[:] #%
            sfc_pres = ds.sp.sel(time=request_dt, method='nearest').data #hPa
            
    elif source == "era5":
        #build file paths
        month_str = str(request_dt.month).zfill(2)
        year_str = str(request_dt.year)
        temp_ffn = glob(f'{era5_pl_root}/t/{year_str}/t_era5_oper_pl_{year_str}{month_str}*.nc')[0]
        geop_ffn = glob(f'{era5_pl_root}/z/{year_str}/z_era5_oper_pl_{year_str}{month_str}*.nc')[0]
        rh_ffn   = glob(f'{era5_pl_root}/r/{year_str}/r_era5_oper_pl_{year_str}{month_str}*.nc')[0]
        sp_ffn   = glob(f'{era5_sl_root}/sp/{year_str}/sp_era5_oper_sfc_{year_str}{month_str}*.nc')[0]

        #extract data
        with xr.open_dataset(temp_ffn) as temp_ds:
            temp_profile = temp_ds.t.sel(longitude=request_lon, method='nearest').sel(latitude=request_lat, method='nearest').sel(time=request_dt, method='nearest').data[:] - 273.15 #units: deg K -> C
        with xr.open_dataset(geop_ffn) as geop_ds:
            geopot_profile = geop_ds.z.sel(longitude=request_lon, method='nearest').sel(latitude=request_lat, method='nearest').sel(time=request_dt, method='nearest').data[:]/9.80665 #units: m**2 s**-2 -> m
            pres_profile = geop_ds.level.data[:] #units: hpa
        with xr.open_dataset(rh_ffn) as rh_ds:
            rh_profile = rh_ds.r.sel(longitude=request_lon, method='nearest').sel(latitude=request_lat, method='nearest').sel(time=request_dt, method='nearest').data[:] #units: percentage
        with xr.open_dataset(sp_ffn) as sp_ds:
            sfc_pres = sp_ds.sp.sel(longitude=request_lon, method='nearest').sel(latitude=request_lat, method='nearest').sel(time=request_dt, method='nearest').data/100 #units: Pa 
        
    #remove levels below the surface pressure
    valid_mask     = np.logical_and(pres_profile < sfc_pres, geopot_profile > request_alt)
    pres_profile   = pres_profile[valid_mask]    
    temp_profile   = temp_profile[valid_mask]    
    geopot_profile = geopot_profile[valid_mask]     
    rh_profile     = rh_profile[valid_mask]
    
    #flipdata (ground is first row)
    if geopot_profile[0]>geopot_profile[-1]:
        temp_profile = np.flipud(temp_profile)
        geopot_profile = np.flipud(geopot_profile)   
        pres_profile = np.flipud(pres_profile)   
        rh_profile = np.flipud(rh_profile)
    
    #append surface data using lowest level
    geopot_profile = np.append(request_alt, geopot_profile)
    pres_profile = np.append(sfc_pres, pres_profile)
    temp_profile = np.append(temp_profile[0], temp_profile) #TODO: use screen temp/RH
    rh_profile = np.append(rh_profile[0], rh_profile)
    
    
    #map temp and z to radar gates
    z_field, temp_field = pyart.retrieve.map_profile_to_gates(temp_profile, geopot_profile, radar) #geopot and temp profile with respect to MSL
    temp_info_field = {'data': temp_field['data'],  # Switch to celsius.
                      'long_name': f'dry bulb temperature',
                      'valid_min': -100, 'valid_max': 100,
                      'units': 'degrees Celsius',
                      'description': f'derived using pressure level temperature and geopotential from {source}',
                      'comment': 'model file timestamp: %s' % (request_dt.strftime("%Y%m%d_%H%M%S"))}
    z_field['description'] = f'derived using pressure level temperature and geopotential from {source}'
    z_field['comment'] = 'model file timestamp: %s' % (request_dt.strftime("%Y%m%d_%H%M%S"))
    
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
                      'long_name': 'Height relative to melting level',
                      'description': 'melting level defined as (Height of 0C level + Height of 10C level)/2 as desribed by Wang et al. 2019 doi:10.1175/JHM-D-18-0071.1',
                      'units': 'm'}
    
    #interpolate to 0C and -20C levels
    fz_level = np.round(sounding_interp(temp_profile, geopot_profile, 0))
    minus_20_level = np.round(sounding_interp(temp_profile, geopot_profile, -20))

    #interpolate to wbt 0C and wbt -20C levels
    wbt_profile = wbt(temp_profile, rh_profile)
    wbt_minus25C = np.round(sounding_interp(wbt_profile, geopot_profile, -25))
    wbt_0C = np.round(sounding_interp(wbt_profile, geopot_profile, 0))

    # store levels in level dictionary
    levels_dict = {'fz_level':fz_level, 'minus_20_level':minus_20_level,
                   'wbt_minus_25_level':wbt_minus25C, 'wbt_0_level':wbt_0C}
    
    #insert original profiles into a dictionary
    profile_dict = {'t':temp_profile, 'z':geopot_profile, 'r':rh_profile, 'p':pres_profile}

    return z_field, temp_info_field, isom_field, profile_dict, levels_dict

def sounding_interp(snd_temp, snd_z, target_temp):
    """
    Provides an linear interpolated height/pressure for a target temperature using a
    sounding vertical profile. Looks for first instance of temperature
    below target_temp from surface upward.

    Parameters
    ----------
    snd_temp : ndarray
        Temperature data (degrees C).
    snd_z : ndarray
        Relative height data (m) or pressure (hpa).
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
        #print(f'warning, target temp {target_temp} warmer than lowest level in sounding {snd_temp[0]}, returning lowest level in sounding')
        return snd_z[0]
    
    # find index above and below freezing level
    mask = np.where(snd_temp < target_temp)
    above_ind = mask[0][0]

    # index below
    below_ind = above_ind - 1
    
    # apply linear interplation to points above and below target_temp
    set_interp = interp1d(
        snd_temp[below_ind:above_ind+1],
        snd_z[below_ind:above_ind+1], kind='linear')
    
    # apply interpolant
    intp_h = set_interp(target_temp)
    
    return intp_h

def find_melting_level(temp_profile, geop_profile):
    #interpolate to required levels
    plus10_h = sounding_interp(temp_profile, geop_profile, 10.)
    fz_h = sounding_interp(temp_profile, geop_profile, 0.)
    #calculate base of melting level
    return (plus10_h+fz_h)/2

def wbt(temp, rh):
    """
    Calculate wet bulb temperature from temperature and relative humidity.

    Parameters
    ----------
    temp : ndarray
        Temperature data (degrees C).
    rh : ndarray
        Relative humidity data (%).

    Returns
    -------
    wb_temp : ndarray
        Wet bulb temperature (degrees C).

    """
    wb_temp = (
        temp * np.arctan(0.151977 * (rh + 8.313659) ** 0.5)
        + np.arctan(temp + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * (rh ** 1.5) * np.arctan(0.023101 * rh)
        - 4.686035
    )
    return wb_temp

