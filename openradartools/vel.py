from unravel import dealias
import numpy as np
import h5py

import pyart

def unravel_velocity(radar, gatefilter, vel_name="VEL", dbz_name="DBZ", nyquist=None):
    """
    Unfold Doppler velocity using Py-ART region based algorithm. Automatically
    searches for a folding-corrected velocity field.
    Parameters:
    ===========
    radar:
        Py-ART radar structure.
    gatefilter:
        Filter excluding non meteorological echoes.
    vel_name: str
        Name of the (original) Doppler velocity field.
    dbz_name: str
        Name of the reflecitivity field.
    nyquist: float
        Nyquist velocity co-interval.
    Returns:
    ========
    vel_meta: dict
        Unfolded Doppler velocity.
    """

    unfvel = dealias.unravel_3D_pyart(
        radar, vel_name, dbz_name,
        gatefilter=gatefilter,
        nyquist_velocity=nyquist,
        do_3d=False
    )
    #build mask
    invalid_mask = np.logical_or(np.isnan(unfvel), gatefilter.gate_excluded)
    #remove nan (for gridding)
    unfvel[np.isnan(unfvel)] = 0.0
    #apply mask
    unfvel = np.ma.masked_where(invalid_mask, unfvel).astype(np.float32)
    
    vel_meta = pyart.config.get_metadata("velocity")
    vel_meta["data"] = unfvel
    vel_meta["_FillValue"] = -9999
    vel_meta["comment"] = "Corrected using the UNRAVEL algorithm developed by Louf et al. (2020) doi:10.1175/jtech-d-19-0020.1 available at https://github.com/vlouf/dealias"
    vel_meta["long_name"] = "Doppler radial velocity of scatterers away from instrument"
    vel_meta["units"] = "m s-1"

    return vel_meta


def extract_nyquist(radar, odim_ffn):
    """
    Extracts the nyquist value for each sweep
    Parameters:
    ===========
    radgrid: struct
        Py-ART grid object.
    odim_ffn: string
        Path of odim full filename
    Returns:
    ========
    nyquist_list: list
        nyquist value for each sweep
    """
    #build list of nyquist for each file
    nyquist_list = []
    #extract nyquist from tilts (as not present in radar object)
    with h5py.File(odim_ffn, 'a') as hfile:
        for i in range(0, radar.nsweeps):
            ds_index = i+1
            try:
                #first look in dataset#/how/NI
                ds_how = hfile['dataset' + str(ds_index)]['how'].attrs
                nyquist_vel = ds_how['NI']
            except:
                try:
                    #second try to calculate from prf and wavelength
                    global_how = hfile['how'].attrs
                    wavelength = global_how['wavelength']/100
                    ds_how = hfile['dataset' + str(ds_index)]['how'].attrs
                    highprf = ds_how['highprf']
                    try:
                        #dual PRF
                        lowprf = ds_how['lowprf']
                        n_folding = lowprf/(highprf-lowprf)
                    except:
                        #single PRF
                        n_folding = 1
                    nyquist_vel = highprf*(wavelength/4)*n_folding
                except:
                    try:
                        #next try dataset#/data2/what/offset
                        ds_data2_what = hfile['dataset' + str(ds_index)]['data2']['what'].attrs
                        if ds_data2_what['quantity'] == 'VRADH':
                            nyquist_vel = round(abs(ds_data2_what['offset']),1)
                        else:
                            print('!!!!!!!!!!!!!!NI for tilt',ds_index,'missing, aborting velocity processing')
                            return []
                        #print('using secondary nqyuist of', nyquist_vel)
                    except:
                        print('!!!!!!!!!!!!!!NI for tilt',ds_index,'missing, aborting velocity processing')
                        #abort, can't handle missing nyquist information
                        return []
                #write back to file as missing nyquist field (used by dualPRF module)
                ds_how['NI'] = nyquist_vel
            #append to list
            nyquist_list.append(nyquist_vel)

    return nyquist_list