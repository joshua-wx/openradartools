import numpy as np
import scipy.ndimage.interpolation as ip
import scipy.optimize as op
from scipy.ndimage import map_coordinates
from skimage.filters import gaussian

#from pysteps
def constant_advection(R, **kwargs):
    """Compute a constant advection field by finding a translation vector that
    maximizes the correlation between two successive images.
    Parameters
    ----------
    R : array_like
      Array of shape (T,m,n) containing a sequence of T two-dimensional input
      images of shape (m,n). If T > 2, two last elements along axis 0 are used.
    Returns
    -------
    out : array_like
        The constant advection field having shape (2, m, n), where out[0, :, :]
        contains the x-components of the motion vectors and out[1, :, :]
        contains the y-components.
    """
    m, n = R.shape[1:]
    X, Y = np.meshgrid(np.arange(n), np.arange(m))

    def f(v):
        XYW = [Y + v[1], X + v[0]]
        R_w = ip.map_coordinates(
            R[-2, :, :], XYW, mode="constant", cval=-1, order=0, prefilter=False
        )

        mask = np.logical_and(np.isfinite(R[-1, :, :]), np.isfinite(R_w))

        return -np.corrcoef(R[-1, :, :][mask], R_w[mask])[0, 1]

    options = {"initial_simplex": (np.array([(0, 1), (1, 0), (1, 1)]))}
    result = op.minimize(f, (1, 1), method="Nelder-Mead", options=options)

    return np.stack([-result.x[0] * np.ones((m, n)), -result.x[1] * np.ones((m, n))])

# #from pysteps
# def advection_correction(R, V, T=5, t=1, mode='max'):
#     """
#     R = np.array([qpe_previous, qpe_current])
#     V = np.array of optical flow vectors [x,y]
#     T = time between two observations (5 min)
#     t = interpolation timestep (1 min)
#     """
    
#     # Perform temporal interpolation
#     Rd = np.zeros((R[0].shape))
#     x, y = np.meshgrid(
#         np.arange(R[0].shape[1], dtype=float), np.arange(R[0].shape[0], dtype=float),
#     )
#     for i in range(t, T + t, t):

#         pos1 = (y - i / T * V[1], x - i / T * V[0])
#         R1 = map_coordinates(R[0], pos1, order=1)

#         pos2 = (y + (T - i) / T * V[1], x + (T - i) / T * V[0])
#         R2 = map_coordinates(R[1], pos2, order=1)

#         Rd_temp = np.amax(np.stack((R1, R2), axis=2), axis=2)
        
#         if mode == 'max':
#             #take the maximum when accumulating the swath
#             Rd = np.amax(np.stack((Rd, Rd_temp), axis=2), axis=2)
#         elif mode == 'sum':
#             #take the sum when accumulating the swath
#             Rd = np.sum(np.stack((Rd, Rd_temp), axis=2), axis=2)
#         else:
#             print('unknown mode')
#     return Rd

#from pysteps
def advection_nowcast(R, V, t=1, T=5, T_end=30, mode='max', fill=0, round_R=False):
    """
    R = np.ma.array of qpe_current
    V = np.ma.array of optical flow vectors [x,y]
    T = time between two observations (5 min)
    end_T = end time for nowcast
    t = interpolation timestep (1 min)
    """
    
    # Perform temporal interpolation
    Rd = np.ma.zeros((R.shape))
    x, y = np.meshgrid(
        np.arange(R.shape[1], dtype=float), np.arange(R.shape[0], dtype=float),
    )
    #smooth R
    Rsmooth = gaussian(R.filled(fill), sigma=1)
    if round_R:
        Rsmooth = np.round(Rsmooth)
    
    for i in np.arange(t, (T_end/T) + t, t):
        
        #dilated_R = dilation(dilated_R, selem=np.ones((3,3)))
        
        ts_forward = -i
        y_shift = y + (ts_forward * V[1])
        x_shift = x + (ts_forward * V[0])
        pos = (y_shift, x_shift)
        R_new = map_coordinates(Rsmooth, pos, order=1)
        R_new = np.ma.masked_array(R_new, R.mask)
        if mode == 'max':
            #take the maximum when accumulating the swath
            Rd = np.ma.max(np.ma.stack((Rd, R_new), axis=2), axis=2, fill_value=0)
        elif mode == 'sum':
            #take the sum when accumulating the swath
            Rd = np.ma.sum(np.ma.stack((Rd, R_new), axis=2), axis=2)
        else:
            print('unknown mode')
            
    return np.ma.masked_array(Rd, Rd==0)


def advection(R,
              V,
              T_start=0, T_end=6,
              T=6, t=1,
              mode='max'):
    """
    WHAT:
    Applies the optical flow from data 1 and data 2 at an interval of t.
    T_start and T_stop allow the user to advect across a portion of the time between the datasets
    By setting the mode to max and sum, different accumulations can be achived.
    
    HELP
    note: first radar volume is the oldest. second radar volume is the newest.

    INPUTS
    R: list of ndarray. intensity values from first radar volume and second radar volume
    oflow_pix: optical flow (pix/min) in [u and v] between the first radar volume and second radar volume
    T_start: starting timestep (minutes from radar volume 1) for interpolation (0 will include the first radar timestep)
    T_end: ending timestep (minutes from radar volume 2) for interpolation (T_end=T will include the second radar timestep)
    T: time difference between radar volumes (minutes)
    t: timestep for interpolation (minutes)
    
    OUTPUTS:
    r_acc: accumulated rainfall totals (mm)
    """

    
    
    # Perform temporal interpolation
    Rd = np.ma.zeros((R[0].shape))
    x, y = np.meshgrid(
        np.arange(Rd.shape[1], dtype=float), np.arange(Rd.shape[0], dtype=float)
    )
    for i in range(T_start, T_end + t, t):
        #shift timestep 1 forwards (this is the older timestep)
        pos1 = (y - i / T * V[1], x - i / T * V[0])
        R1 = map_coordinates(R[0], pos1, order=1)
        R1 = np.ma.masked_array(R1, R[0].mask)
        weight1 = (T - i)/T

        #shift timestep 2 backwards (this is the newer timestep)
        pos2 = (y + (T - i) / T * V[1], x + (T - i) / T * V[0])
        R2 = map_coordinates(R[1], pos2, order=1)
        R2 = np.ma.masked_array(R2, R[1].mask)
        weight2 = i/T
        
        #weighted combination
        R_new = R1 * weight1 + R2 * weight2
        #convert to mm/hr in mm
        if mode == 'sum':
            Rd = np.ma.sum(np.ma.stack((Rd, R_new), axis=2), axis=2) #ma assigned to 0
        elif mode == 'max':
            Rd = np.ma.max(np.ma.stack((Rd, R_new), axis=2), axis=2, fill_value=0) #ma assigned to 0
        else:
            print('unknown mode')
    
    return np.ma.masked_array(Rd, Rd==0)