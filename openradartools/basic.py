from datetime import datetime, timedelta
import statistics
import numpy as np

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

def list_to_dt(dt_list, str_format):
    """
    Convert a list of string to a list of datetimes using the str_format

    Parameters
    ----------
    dt_list : list
        list of time/date strings
    str_format : string
        datetime format string

    Returns
    -------
    dt_list : list
        list of datetime values

    """
    
    dt_list = []
    for item in dt_list:
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