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


def open_dem(dem_fn='australia_250m_dem.tif', invalid_terrain=-9999):
    """
    Open a DEM file for generating masks. returns variables required for wradlib processing
    """
    rasterfile = wrl.util.get_wradlib_data_file(dem_fn)
    ds = wrl.io.open_raster(rasterfile)
    rastervalues, rastercoords, proj = wrl.georef.extract_raster_dataset(ds, nodata=invalid_terrain)
    
    return (rastervalues, rastercoords, proj)

def build_masks(vol_ffn, dem_info, bw_peaks=3.0, invalid_terrain = -9999, terrain_offset = 2000):
    
    #build radar info
    radar = pyart.aux_io.read_odim_h5(vol_ffn)
    sitecoords = (radar.longitude['data'][0], radar.latitude['data'][0], radar.altitude['data'][0])
    nrays = int(radar.nrays/radar.nsweeps) # number of rays
    nbins = radar.ngates # number of range bins
    el_list = radar.fixed_angle['data'] # vertical antenna pointing angle (deg)
    range_res = radar.range['data'][2] - radar.range['data'][1]# range resolution (meters)
    
    #unpack DEM
    rastervalues, rastercoords, proj = dem_info

    #build coordiantes
    coord = None
    for el in el_list:
        #calculat spherical coordiantes for a sweep
        sweep_coord = wrl.georef.sweep_centroids(nrays, range_res, nbins, el)
        if coord is None:
            coord = sweep_coord
        else:
            #append spherical coordiantes for a sweep
            coord = np.append(coord, sweep_coord, axis=0)
    #calculate geographical coordinates of spherical space
    coords = wrl.georef.spherical_to_proj(coord[..., 0],
                                          coord[..., 1],
                                          coord[..., 2], sitecoords)
    lon = coords[..., 0]
    lat = coords[..., 1]
    alt = coords[..., 2]
    #polar coodinates for mapping terrain (no altitude)
    polcoords = coords[..., :2]

    # Clip the region inside our bounding box
    rlimits = (lon.min(), lat.min(), lon.max(), lat.max())
    ind = wrl.util.find_bbox_indices(rastercoords, rlimits)
    rastercoords_clip = rastercoords.copy()[ind[1]:ind[3], ind[0]:ind[2], ...]
    rastervalues_clip = rastervalues.copy()[ind[1]:ind[3], ind[0]:ind[2]]

    # Map rastervalues to polar grid points
    polarvalues = wrl.ipol.cart_to_irregular_interp(rastercoords_clip, rastervalues_clip,
                                                 polcoords, method='nearest')
    
    #calculate sea mask using invalid terrain value
    sea_mask = polarvalues == invalid_terrain

    #calculate clutter mask
    #where beam centre + 3dB beam width is lower than the terrain + terrain_offset
    r = np.arange(nbins) * range_res
    beamradius = wrl.util.half_power_radius(r, bw_peaks)
    beam_bottom = alt - beamradius
    clutter_mask = beam_bottom <= (polarvalues + terrain_offset)
    
    return sea_mask, clutter_mask
