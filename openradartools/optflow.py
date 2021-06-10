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
def advection_correction(R, V, T=5, t=1):
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
        
        Rd = np.amax(np.stack((Rd, Rd_temp), axis=2), axis=2)

    return Rd