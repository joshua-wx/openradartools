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