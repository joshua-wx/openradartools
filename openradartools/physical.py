import os

import numpy as np
import wradlib as wrl

def open_dem(dem_fn='australia_250m_dem.tif', invalid_terrain=-9999):
    """
    Open a DEM file for generating masks. returns variables required for wradlib processing
    """
    rasterfile = wrl.util.get_wradlib_data_file(dem_fn)
    ds = wrl.io.open_raster(rasterfile)
    rastervalues, rastercoords, proj = wrl.georef.extract_raster_dataset(ds, nodata=invalid_terrain)
    
    return (rastervalues, rastercoords, proj)

def beam_blocking(radar, radar_id, radar_dt, srtm_ffn, output_root, force=False):
    """
    Apply the wradlib beam blocking library for the target volume.

    Parameters
    ----------
    radar : Radar
        Py-ART radar object.
    srtm_ffn : string
        Full path to SRTM geotiff file.

    Returns
    -------
    ccb_dict : dict
        Dictionary containing the cumulative beam blocking (CBB) for
        every pixel in the radar object.

    """
    #init
    bb_path = f'{output_root}/{radar_id:02}/{radar_dt.strftime("%Y%m%d")}'
    bb_ffn = f'{bb_path}/{radar_id:02}_{radar_dt.strftime("%Y%m%d")}_bb.npz'
    #create paths
    if not os.path.exists(bb_path):
        os.makedirs(bb_path)
        
    #check if BB already calculated
    if os.path.exists(bb_ffn) and force == False:
        print('BB file exists, skipping processing:', bb_ffn)
        data = np.load(bb_ffn)
        CBB = data['CBB']
        
    else:
        # site parameters
        radar_lat = radar.latitude['data'][0]
        radar_lon = radar.longitude['data'][0]
        radar_alt = radar.altitude['data'][0]
        sitecoords = (radar_lon, radar_lat, radar_alt)
        nsweeps = radar.nsweeps
        nrays = int(radar.nrays / nsweeps)
        nbins = int(radar.ngates)
        el_list = radar.fixed_angle['data']
        range_res = radar.range['meters_between_gates']
        try:
            bw = radar.instrument_parameters['radar_beam_width_h']['data']
        except:
            print('beamwidth info missing form volume, using default of 1deg')
            bw = 1
            
        # grid arrays
        r = np.arange(nbins) * range_res
        beamradius = wrl.util.half_power_radius(r, bw)

        # read geotiff
        ds = wrl.io.open_raster(srtm_ffn)
        rastervalues, rastercoords, proj = wrl.georef.extract_raster_dataset(ds, nodata=-32768)

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

        # calculate beam blocking for each bin
        PBB = wrl.qual.beam_block_frac(polarvalues, alt, beamradius)
        PBB = np.ma.masked_invalid(PBB)

        # calculate beam blocking along each ray
        CBB = wrl.qual.cum_beam_block_frac(PBB)

        # save to npz file
        np.savez(bb_ffn, CBB=CBB, PBB=PBB)

    # generate meta
    the_comments = "wradlib cumulative beam blocking"
    cbb_dict = {'data': CBB, 'units': '%',
                'long_name': 'cumulative beam blocking percentage',
                'standard_name': 'CBB', 'comments': the_comments}

    return cbb_dict

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
