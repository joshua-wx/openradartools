import os
import time
import numpy as np
import pyart

import wradlib as wrl
import wradlib.clutter as clutter
os.environ["WRADLIB_DATA"] = "/g/data1a/kl02/jss548/GIS_data"

def _filter_hardcoding(my_array, nuke_filter, bad=-9999):
    """
    Harcoding GateFilter into an array.
    Parameters:
    ===========
        my_array: array
            Array we want to clean out.
        nuke_filter: gatefilter
            Filter we want to apply to the data.
        bad: float
            Fill value.
    Returns:
    ========
        to_return: masked array
            Same as my_array but with all data corresponding to a gate filter
            excluded.
    """
    filt_array = np.ma.masked_where(nuke_filter.gate_excluded, my_array.copy())
    filt_array = filt_array.filled(fill_value=bad)
    to_return = np.ma.masked_where(filt_array == bad, filt_array)
    return to_return    

def apply_gpmmatch_calibration(radar, radar_dt, cal_dict, in_dbz_name, out_dbz_name):
    #find refl cal value
    error_msg = ''
    dbzh_offset = 0
    if cal_dict:
        caldt_mask  = np.logical_and(radar_dt.date()>=cal_dict['cal_start'], radar_dt.date()<=cal_dict['cal_end'])
        dbzh_offset_match = cal_dict['cal_mean'][caldt_mask]
        if len(dbzh_offset_match) == 0:
            error_msg = 'time period not found in cal file'
        elif len(dbzh_offset_match) > 1:
            error_msg = 'multiple matches found in cal file'
            print('calibration data error (multiple matches)')                
        else:
            dbzh_offset = float(dbzh_offset_match)
    else:
        error_msg = 'no cal file found'
    
    #apply calibration
    refl_cal_data     = radar.fields[in_dbz_name]['data'].copy() - dbzh_offset #msgr is GR-SR
    radar.add_field_like(in_dbz_name, out_dbz_name, refl_cal_data)
    radar.fields[out_dbz_name]['calibration_offset'] = dbzh_offset
    radar.fields[out_dbz_name]['calibration_units'] = 'dBZ'
    if len(error_msg) == 0:
        radar.fields[out_dbz_name]['calibration_notes'] = 'GR_cal = GR - cal_offset'
    else:
        radar.fields[out_dbz_name]['calibration_notes'] = error_msg
    return radar

def clean_sp(radar, tilt_list, in_dbz_name, out_dbz_name):
    """
    Apply a clutter removal workflow to a single radar volume
    ===============
    (1) Gabella texture filter
    (2) Despeckle
    Parameters:
    ===============
        radar: pyart radar object
        
        tilt_list: list
            tilt list to process. Empty triggers processing for all tilts
    Returns:
    ===============
        radar: pyart radar object
    """

    #config
    rain_cut_dbz  = 10.
    #min_dbz       = 0.
    #snr_threshold = 10
    
    #define the indices for the required sweep
    sweep_startidx = radar.sweep_start_ray_index['data'][:]
    sweep_endidx   = radar.sweep_end_ray_index['data'][:]
    refl_data      = radar.fields[in_dbz_name]['data'].copy()
    clutter_mask   = np.zeros_like(refl_data) #clutter flag = 1, no clutter = 0

    #build list of tilts to process for gabella filter
    if not tilt_list:
        tilt_list = np.arange(len(sweep_startidx))
    else:
        tilt_list = np.array(tilt_list)
        
    #loop through sweeps    
    for k in tilt_list:
        #extract ppi
        ppi            = refl_data[sweep_startidx[k]:sweep_endidx[k]+1]
        #generate clutter mask for ppi
        clmap = clutter.filter_gabella(ppi,
                                       wsize=5,
                                       thrsnorain=rain_cut_dbz,
                                       tr1=10.,
                                       n_p=8,
                                       tr2=1.3)
        #insert clutter mask for ppi into volume mask
        clutter_mask[sweep_startidx[k]:sweep_endidx[k]+1] = clmap

    #add clutter mask to radar object
    clutter_field = {'data': clutter_mask, 'units': 'mask', 'long_name': 'Gabella clutter mask',
                            'standard_name': 'CLUTTER', 'comments': 'wradlib implementation of Gabella et al., 2002'}
    radar.add_field('reflectivity_clutter', clutter_field, replace_existing=True) 
    
    #apply clutter filter to gatefiler
    gatefilter = pyart.correct.GateFilter(radar)
    gatefilter.exclude_equal('reflectivity_clutter',1)
    
    #generate depseckle filter
    gate_range       = radar.range['meters_between_gates']
    #apply limits
    if gate_range < 250:
        print('gate range too small in clutter.py, setting to 250m')
        gate_range = 250
    if gate_range > 1000:
        print('gate range too large in clutter.py, setting to 1000m')
        gate_range = 1000
    #rescale and calculate sz
    despeckle_factor = 1000/gate_range #scale despeckle according to gate size 
    despeckle_sz     = 15 * despeckle_factor #1000m = 15, 500m = 30, 250m = 60
    #apply despeckle to gatefilter
    gatefilter       = pyart.correct.despeckle.despeckle_field(radar, in_dbz_name, gatefilter=gatefilter, size=despeckle_sz)
    
    #apply filter to mask
    cor_refl_data = np.ma.masked_where(gatefilter.gate_excluded, refl_data.copy())

    #update radar object
    radar.add_field_like(in_dbz_name, out_dbz_name, cor_refl_data.astype(np.float32), replace_existing=True)
    radar.fields[out_dbz_name]['_FillValue'] = -9999
    radar.fields[out_dbz_name]['comment'] = 'Corrected for attenuation (C band only), absolute calibration and clutter removal'
    radar.fields[out_dbz_name]['long_name'] = 'Horizontal Reflectivity'
    radar.fields[out_dbz_name]['standard_name'] = 'horizontal_reflectivity'
    radar.fields[out_dbz_name]['units'] = 'dBZ'
    
    #return radar object
    return radar, gatefilter

def c_band_attenuation(radar, radar_band, in_dbz_name, out_dbz_name, minimum_range=10):
    """
    Apply C band reflectivity correction for single pol from https://github.com/vlouf/gpmmatch/blob/master/gpmmatch/correct.py
    """
    #extract data
    refl_data = radar.fields[in_dbz_name]['data'].copy()
    range_dim = radar.range['data']/1000
    min_range_indx = np.where(range_dim>=minimum_range)[0][0]
    
    #apply only for c band
    if radar_band == 'C':
        ze = 10 ** (refl_data.copy() / 10)
        ze[:, 0:min_range_indx] = 0 #mask for less than minimum range
        atten = 1.31885e-6 * ze + 1.8041e-3
        refl_data = refl_data + 2 * np.cumsum(atten, axis=1)
        radar.add_field_like(in_dbz_name, out_dbz_name, refl_data, replace_existing=True)
        radar.fields[out_dbz_name]['sp_attenuation_correction'] = f'C band correction for single polarisation. Minimum range {minimum_range}km'
    else:
        radar.add_field_like(in_dbz_name, out_dbz_name, refl_data, replace_existing=True)
        radar.fields[out_dbz_name]['sp_attenuation_correction'] = 'None'
        
    return radar