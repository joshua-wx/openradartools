from datetime import datetime, timedelta
import statistics
import numpy as np

import pyart

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    From http://stackoverflow.com/a/312464
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
def daterange(date1, date2):
    """
    Generate date list between dates
    """
    date_list = []
    for n in range(int ((date2 - date1).days)+1):
        date_list.append(date1 + timedelta(n))
    return date_list

def calc_step(dt_list):
    start_dt_list = dt_list[:-1]
    end_dt_list   = dt_list[1:]
    t_diff        = np.zeros(len(end_dt_list))
    for i,_ in enumerate(t_diff):
        t_diff[i] = round((end_dt_list[i] - start_dt_list[i]).total_seconds() / 60.0)
    try:
        mode_step = statistics.mode(t_diff)
    except Exception as e:
        print('mode step failed with', e, 'defaulting to 10min')
        mode_step = 10
    #enforce known steps
    if mode_step not in [5,6,10]:
        print('Mode step failed, defaulting to 10min')
        mode_step = 10
    
    return mode_step

def list_to_dt(str_list, str_format):
    """
    Convert a list of string to a list of datetimes using the str_format

    Parameters
    ----------
    str_list : list
        list of time/date strings
    str_format : string
        datetime format string

    Returns
    -------
    dt_list : list
        list of datetime values

    """
    
    dt_list = []
    for item in str_list:
        dt_list.append(datetime.strptime(item, str_format))
        
    return dt_list
    

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
    wb_temp = (temp * np.arctan(0.151977*(rh+8.313659)**0.5)
               + np.arctan(temp+rh) - np.arctan(rh-1.676331)
               + 0.00391838*(rh**1.5)*np.arctan(0.023101*rh) - 4.686035)
    return wb_temp

def cfradial_to_3dgrid(radar, field_name='reflectivity'):
    
    """
    Converts a radar field from the 2D cfradial representation to a 3D grid
    with dimensions azimuth, range and elevation.
    Also sorts out elevation so the first elevation is the lowest elevation.
    
    INPUTS:
        radar (pyart radar object)
        field_name (string)
            name of field in pyart object
    OUTPUTS:
        data_grid (np.ma.array)
            Output 3D array
        coordinates (dictionary)
            coordinate vector of 3D array in their order of representation
    """

    #sorted elevation
    el_sort_idx = np.argsort(radar.fixed_angle['data'])
    az = radar.get_azimuth(0)
    rg = radar.range['data']

    #init 3D grid
    data_grid = np.ma.zeros((len(az), len(rg), len(el_sort_idx)))

    #insert sweeps into 3D grid
    for i, el_idx in enumerate(el_sort_idx):
        data_grid[:, :, i] = radar.get_field(el_idx, field_name)

    #create dictionary for coordinates (in order)
    coordinates = {'azimuth':az, 'range':rg, 'elevation':radar.elevation['data'][el_sort_idx]}

    return data_grid, coordinates

def get_radar_z(radar):
    
    """
    uses the pyart antenna_to_cartesian function to derive the 
    altitude of each radar sample above the ground (using the radar altitude)
    
    INPUTS:
        radar (pyart radar object)
    OUTPUTS:
        height_dict (dictionary)
            pyart field with height data and necessary metadata (defaults)
    
    """
    # retrieve the Z coordinates of the radar gates
    az = radar.get_azimuth(0)
    rg = radar.range['data']
    el = radar.fixed_angle['data']
    
    rg_grid, az_grid = np.meshgrid(rg, az)
    rg_grid, el_grid = np.meshgrid(rg, el)
    _, _, z = pyart.core.transforms.antenna_to_cartesian(rg_grid / 1000.0, rg, az)
    # Check that z is not a MaskedArray
    if isinstance(z, np.ma.MaskedArray):
        z = z.filled(np.NaN)
    
    height_field = pyart.config.get_field_name('height')
    height_dict = pyart.config.get_metadata(height_field)
    height_dict['data'] = z + radar.altitude['data'][0]

    
    return height_dict