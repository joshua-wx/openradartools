import numpy as np
import scipy.ndimage.interpolation as ip
import scipy.optimize as op
from scipy.ndimage import map_coordinates

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

#from pysteps
def advection_correction(R, V, T=5, t=1, mode='max'):
    """
    R = np.array([qpe_previous, qpe_current])
    V = np.array of optical flow vectors [x,y]
    T = time between two observations (5 min)
    t = interpolation timestep (1 min)
    """
    
    # Perform temporal interpolation
    Rd = np.zeros((R[0].shape))
    x, y = np.meshgrid(
        np.arange(R[0].shape[1], dtype=float), np.arange(R[0].shape[0], dtype=float),
    )
    for i in range(t, T + t, t):

        pos1 = (y - i / T * V[1], x - i / T * V[0])
        R1 = map_coordinates(R[0], pos1, order=1)

        pos2 = (y + (T - i) / T * V[1], x + (T - i) / T * V[0])
        R2 = map_coordinates(R[1], pos2, order=1)

        Rd_temp = np.amax(np.stack((R1, R2), axis=2), axis=2)
        
        if mode == 'max':
            #take the maximum when accumulating the swath
            Rd = np.amax(np.stack((Rd, Rd_temp), axis=2), axis=2)
        elif mode == 'sum':
            #take the sum when accumulating the swath
            Rd = np.sum(np.stack((Rd, Rd_temp), axis=2), axis=2)
        else:
            print('unknown mode')
    return Rd


# def advection(data1, data2,
#               oflow1_pix, oflow2_pix,
#               T_start=0, T_end=6,
#               T=6, t=1,
#               mode='max'):
#     """
#     WHAT:
#     Applies the optical flow from data 1 and data 2 at an interval of t.
#     T_start and T_stop allow the user to advect across a portion of the time between the datasets
#     By setting the mode to max and sum, different accumulations can be achived.
    
#     HELP
#     note: first radar volume is the oldest. second radar volume is the newest.

#     INPUTS
#     rrate1/rrate2: rain rate (mm/hr) for first radar volume and second radar volume
#     oflow1_pix/oflow2_pix: optical flow (pix/min) in [u and v] direction for first radar volume and second radar volume (list of length 2, elements are 2D np.array)
#     T_start: starting timestep (minutes from radar volume 1) for interpolation (0 will include the first radar timestep)
#     T_end: ending timestep (minutes from radar volume 2) for interpolation (T_end=T will include the second radar timestep)
#     T: time difference between radar volumes (minutes)
#     t: timestep for interpolation (minutes)
    
#     OUTPUTS:
#     r_acc: accumulated rainfall totals (mm)
#     """

#     # Perform temporal interpolation
#     r_acc = np.zeros((data1.shape))
#     x, y = np.meshgrid(
#         np.arange(rrate1.shape[1], dtype=float), np.arange(rrate1.shape[0], dtype=float)
#     )
#     for i in range(T_start, T_end + t, t):
#         #shift timestep 1 forwards (this is the older timestep)
#         ts_forward = -i
#         y1_shift = y + (ts_forward * oflow1_pix[1])
#         x1_shift = x + (ts_forward * oflow1_pix[0])
#         pos1 = (y1_shift, x1_shift)
#         R1 = map_coordinates(rrate1, pos1, order=1)
#         weight1 = (T - i)/T

#         #shift timestep 2 backwards (this is the newer timestep)
#         ts_backwards = (T-i)
#         y2_shift = y + (ts_backwards * oflow2_pix[1])
#         x2_shift = x + (ts_backwards * oflow2_pix[0])
#         pos2 = (y2_shift, x2_shift)
#         R2 = map_coordinates(rrate2, pos2, order=1)
#         weight2 = i/T

#         #weighted combination
#         rrate_interp = R1 * weight1 + R2 * weight2

#         #convert to mm/hr in mm
#         r_acc += rrate_interp/60*t
        
#     return r_acc