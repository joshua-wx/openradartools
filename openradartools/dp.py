import os
from glob import glob

import pyart
import numpy as np
from numba import jit
import pandas as pd
import xarray as xr
import cftime
import h5py
from scipy import integrate
from csu_radartools import csu_kdp, csu_fhc

def det_sys_phase_gf(radar, gatefilter, phidp_field=None, first_gate=30, sweep=0):
    """
    Determine the system phase.

    Parameters
    ----------
    radar : Radar
        Radar object for which to determine the system phase.
    gatefilter : Gatefilter
        Gatefilter object highlighting valid gates.
    phidp_field : str, optional
        Field name within the radar object which represents
        differential phase shift. A value of None will use the default
        field name as defined in the Py-ART configuration file.
    first_gate : int, optional
        Gate index for where to being applying the gatefilter.

    Returns
    -------
    sys_phase : float or None
        Estimate of the system phase. None is not estimate can be made.

    """
    # parse the field parameters
    if phidp_field is None:
        phidp_field = get_field_name('differential_phase')
    phidp = radar.fields[phidp_field]['data'][:, first_gate:]
    first_ray_idx = radar.sweep_start_ray_index['data'][sweep]
    last_ray_idx = radar.sweep_end_ray_index['data'][sweep]
    is_meteo = gatefilter.gate_included[:, first_gate:]
    return _det_sys_phase_gf(phidp, first_ray_idx, last_ray_idx, is_meteo)

def _det_sys_phase_gf(phidp, first_ray_idx, last_ray_idx, radar_meteo):
    """ Determine the system phase, see :py:func:`det_sys_phase`. """
    good = False
    phases = []
    for radial in range(first_ray_idx, last_ray_idx + 1):
        meteo = radar_meteo[radial, :]
        mpts = np.where(meteo)
        if len(mpts[0]) > 25:
            good = True
            msmth_phidp = pyart.correct.phase_proc.smooth_and_trim(phidp[radial, mpts[0]], 9)
            phases.append(msmth_phidp[0:25].min())
    if not good:
        return None
    return np.median(phases)

def apply_zdr_calibration(radar, radar_dt, cal_dict, in_zdr_name, out_zdr_name):
    #find refl cal value
    error_msg = ''
    zdr_offset = 0
    if cal_dict:
        caldt_mask  = np.logical_and(radar_dt.date()>=cal_dict['cal_start'], radar_dt.date()<=cal_dict['cal_end'])
        zdr_offset_match = cal_dict['cal_mean'][caldt_mask]
        if len(zdr_offset_match) == 0:
            error_msg = 'time period not found in cal file'
        elif len(zdr_offset_match) > 1:
            error_msg = 'multiple matches found in cal file'
            print('calibration data error (multiple matches)')                
        else:
            zdr_offset = float(zdr_offset_match)
    else:
        error_msg = 'no cal file found'
        zdr_offset = 0
    
    #apply calibration
    zdr_cal_data     = radar.fields[in_zdr_name]['data'].copy() - zdr_offset
    radar.add_field_like(in_zdr_name, out_zdr_name, zdr_cal_data)
    radar.fields[out_zdr_name]['calibration_offset'] = zdr_offset
    radar.fields[out_zdr_name]['calibration_units'] = 'dB'
    
    if len(error_msg) == 0:
        radar.fields[out_zdr_name]['calibration_notes'] = 'calibration derived from birdbath scan'
    else:
        radar.fields[out_zdr_name]['calibration_notes'] = error_msg
    return radar

def do_gatefilter(radar, gf=None, refl_name='DBZ', phidp_name="PHIDP", rhohv_name='RHOHV_CORR', zdr_name="ZDR", despeckle_field=False):
    """
    Basic filtering function for dual-polarisation data.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        gatefilter:
            Py-ART gatefilter object.
        refl_name: str
            Reflectivity field name.
        rhohv_name: str
            Cross correlation ratio field name.
        ncp_name: str
            Name of the normalized_coherent_power field.
        zdr_name: str
            Name of the differential_reflectivity field.

    Returns:
    ========
        gf_despeckeld: GateFilter
            Gate filter (excluding all bad data).
    """
    # Initialize gatefilter
    if gf is None:
        gf = pyart.correct.GateFilter(radar)
    if despeckle_field:
        # Despeckle
        gf = pyart.correct.despeckle_field(radar, refl_name, gatefilter=gf)

    # Remove obviously wrong data.
    gf.exclude_outside(zdr_name, -6.0, 7.0)
    gf.exclude_outside(refl_name, -20.0, 80.0)
    gf.exclude_below(rhohv_name, 0.7)

    return gf

def correct_attenuation_zphi(
    radar,
    gatefilter,
    band='C',
    refl_field="DBZH",
    phidp_field="PHIDP",
    zdr_field='ZDR',
    temp_field='temperature'):
    
    """
    Correct attenuation on reflectivity using Py-ART tool.
    Parameters:
    ===========
    radar:
        Py-ART radar structure.
    gatefilter:
        Filter excluding non meteorological echoes.
    refl_name: str
        Reflectivity field name.
    kdp_name: str
        KDP field name.
    Returns:
    ========
    zh_corr: array
        Attenuation corrected reflectivity.
    zdr_corr: array
        Attenuation corrected differential reflectivity.
    """

    a_coef, beta, c, d = pyart.correct.attenuation._param_attzphi_table()[band]
    
    _,_,cor_z,_,_,cor_zdr = pyart.correct.calculate_attenuation_zphi(
        radar,
        refl_field=refl_field,
        phidp_field=phidp_field,
        zdr_field=zdr_field,
        temp_field=temp_field,
        a_coef=a_coef, beta=beta, c=c, d=d)
    
    #apply mask
    cor_z = np.ma.masked_invalid(cor_z["data"])
    cor_zdr = np.ma.masked_invalid(cor_zdr["data"])
    cor_z = np.ma.masked_where(gatefilter.gate_excluded, cor_z)
    cor_zdr = np.ma.masked_where(gatefilter.gate_excluded, cor_zdr)
    #set fill values
    np.ma.set_fill_value(cor_z, -9999)
    np.ma.set_fill_value(cor_zdr, -9999)
    #return as float
    return cor_z.astype(np.float32), cor_zdr.astype(np.float32)

def phidp_bringi(radar, gatefilter, phidp_field="PHI_UNF", refl_field='DBZ'):
    """
    Compute PHIDP and KDP Bringi.
    Parameters
    ==========
    radar:
        Py-ART radar data structure.
    gatefilter:
        Gate filter.
    unfold_phidp_name: str
        Differential phase key name.
    refl_field: str
        Reflectivity key name.
    Returns:
    ========
    phidpb: ndarray
        Bringi differential phase array.
    kdpb: ndarray
        Bringi specific differential phase array.
    """
    dz = radar.fields[refl_field]['data'].copy().filled(-9999)
    dp = radar.fields[phidp_field]['data'].copy().filled(-9999)

    # Extract dimensions
    rng = radar.range['data']
    azi = radar.azimuth['data']
    dgate = rng[1] - rng[0]
    [R, A] = np.meshgrid(rng, azi)

    # Compute KDP bringi.
    kdpb, phidpb, _ = csu_kdp.calc_kdp_bringi(dp, dz, R / 1e3, gs=dgate, bad=-9999, thsd=12, window=6.0, std_gate=11)

    # Mask array
    phidpb = np.ma.masked_where(phidpb == -9999, phidpb)
    kdpb = np.ma.masked_where(kdpb == -9999, kdpb)
    
    #fill
    phidpb = fill_phi(phidpb.filled(np.NaN))
    
    #set fill values
    np.ma.set_fill_value(phidpb, -9999)
    np.ma.set_fill_value(kdpb, -9999)
    
    # Get metadata.
    phimeta = pyart.config.get_metadata("differential_phase")
    phimeta['data'] = phidpb
    kdpmeta = pyart.config.get_metadata("specific_differential_phase")
    kdpmeta['data'] = kdpb
    
    return phimeta, kdpmeta

@jit
def fill_phi(phi):
    """
    Small function that propagates phidp values forward along rays to fill gaps
    """
    nx, ny = phi.shape
    for i in range(nx):
        phi_val = 0
        for j in range(ny):
            if np.isnan(phi[i, j]):
                phi[i, j] = phi_val
            else:
                phi_val = phi[i, j]
                
    return phi

def insert_ncar_pid(radar, odim_ffn, refl_name='reflectivity'):

    """
    extracts the NCAR PID from BOM ODIMH5 files into a CFRADIAL-type format and returns
    the dictionary containing this new field with the required metadata
    """
    sweep_shape = np.shape(radar.get_field(0, refl_name))
    pid_volume = None
    with h5py.File(odim_ffn, 'r') as f:
        h5keys = list(f.keys())
        #init 
        if 'how' in h5keys:
            h5keys.remove('how')
        if 'what' in h5keys:
            h5keys.remove('what')     
        if 'where' in h5keys:
            h5keys.remove('where')
        n_keys = len(h5keys)

        #collate padded sweeps into a volume
        for i in range(n_keys):
            ds_name = 'dataset' + str(i+1)
            pid_sweep = np.array(f[ds_name]['quality1']['data'])
            shape = pid_sweep.shape
            padded_pid_sweep = np.zeros(sweep_shape)
            padded_pid_sweep[:shape[0],:shape[1]] = pid_sweep
            if pid_volume is None:
                pid_volume = padded_pid_sweep
            else:
                pid_volume = np.vstack((pid_volume, padded_pid_sweep))

        #mask
        pid_volume = np.ma.masked_less_equal(pid_volume.astype(np.int16), 0)
                
    #add to radar object
    the_comments = "0: nodata; 1: Cloud; 2: Drizzle; 3: Light_Rain; 4: Moderate_Rain; 5: Heavy_Rain; " +\
                   "6: Hail; 7: Rain_Hail_Mixture; 8: Graupel_Small_Hail; 9: Graupel_Rain; " +\
                   "10: Dry_Snow; 11: Wet_Snow; 12: Ice_Crystals; 13: Irreg_Ice_Crystals; " +\
                   "14: Supercooled_Liquid_Droplets; 15: Flying_Insects; 16: Second_Trip; 17: Ground_Clutter; " +\
                   "18: misc1; 19: misc2"
    pid_meta = {'data': pid_volume, 'units': ' ', 'long_name': 'NCAR Hydrometeor classification', '_FillValue': np.int16(0),
                  'standard_name': 'Hydrometeor_ID', 'comments': the_comments}

    return pid_meta

def csu_hca(radar, gatefilter, kdp_name, zdr_name, band, refl_name='DBZ_CORR',
                   rhohv_name='RHOHV_CORR',
                   temperature_name='temperature',
                   height_name='height'):
    """
    Compute CSU hydrometeo classification.

    Parameters:
    ===========
    radar:
        Py-ART radar structure.
    refl_name: str
        Reflectivity field name.
    zdr_name: str
        ZDR field name.
    kdp_name: str
        KDP field name.
    rhohv_name: str
        RHOHV field name.
    temperature_name: str
        Sounding temperature field name.
    height: str
        Gate height field name.

    Returns:
    ========
    hydro_meta: dict
        Hydrometeor classification.
    """
    refl = radar.fields[refl_name]['data'].copy().filled(np.NaN)
    zdr = radar.fields[zdr_name]['data'].copy().filled(np.NaN)
    try:
        kdp = radar.fields[kdp_name]['data'].copy().filled(np.NaN)
    except AttributeError:
        kdp = radar.fields[kdp_name]['data'].copy()
    rhohv = radar.fields[rhohv_name]['data']
    try:
        radar_T = radar.fields[temperature_name]['data']
        use_temperature = True
    except Exception:
        use_temperature = False

    if use_temperature:
        scores = csu_fhc.csu_fhc_summer(dz=refl, zdr=zdr, rho=rhohv, kdp=kdp, use_temp=True, band=band, T=radar_T)
    else:
        scores = csu_fhc.csu_fhc_summer(dz=refl, zdr=zdr, rho=rhohv, kdp=kdp, use_temp=False, band=band)

    hydro = np.argmax(scores, axis=0) + 1
    hydro[gatefilter.gate_excluded] = 0
    hydro_data = np.ma.masked_equal(hydro.astype(np.short), 0)

    the_comments = "1: Drizzle; 2: Rain; 3: Ice Crystals; 4: Aggregates; " +\
                   "5: Wet Snow; 6: Vertical Ice; 7: LD Graupel; 8: HD Graupel; 9: Hail; 10: Big Drops"

    hydro_meta = {'data': hydro_data, 'units': ' ', 'long_name': 'CSU Hydrometeor classification', '_FillValue': np.short(0),
                  'standard_name': 'Hydrometeor_ID', 'comments': the_comments}

    return hydro_meta


def correct_rhohv(radar, rhohv_name='RHOHV', snr_name='SNR'):
    """
    Correct cross correlation ratio (RHOHV) from noise. From the Schuur et al.
    2003 NOAA report (p7 eq 5)

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        rhohv_name: str
            Cross correlation field name.
        snr_name: str
            Signal to noise ratio field name.

    Returns:
    ========
        rho_corr: array
            Corrected cross correlation ratio.
    """
    rhohv = radar.fields[rhohv_name]['data'].copy()
    snr = radar.fields[snr_name]['data'].copy()

    #appears to be done onsite at CP2
#     natural_snr = 10**(0.1 * snr)
#     natural_snr = natural_snr.filled(-9999)
#     rho_corr = rhohv * (1 + 1 / natural_snr)
    rho_corr = rhohv
    
    # Not allowing the corrected RHOHV to be lower than the raw rhohv
    rho_corr[np.isnan(rho_corr) | (rho_corr < 0) | (rho_corr > 1)] = 1
    try:
        rho_corr = rho_corr.filled(1)
    except Exception:
        pass

    return np.ma.masked_array(rho_corr, rhohv.mask)

def correct_zdr(radar, zdr_name='ZDR', snr_name='SNR'):
    """
    Correct differential reflectivity (ZDR) from noise. From the Schuur et al.
    2003 NOAA report (p7 eq 6)

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        zdr_name: str
            Differential reflectivity field name.
        snr_name: str
            Signal to noise ratio field name.

    Returns:
    ========
        corr_zdr: array
            Corrected differential reflectivity.
    """
    zdr = radar.fields[zdr_name]['data'].copy()
    snr = radar.fields[snr_name]['data'].copy()
    alpha = 1.48
    natural_zdr = 10 ** (0.1 * zdr)
    natural_snr = 10 ** (0.1 * snr)
    corr_zdr = 10 * np.log10((alpha * natural_snr * natural_zdr) / (alpha * natural_snr + alpha - natural_zdr))

    return corr_zdr