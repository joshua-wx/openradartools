import numpy as np
from scipy.spatial import cKDTree

def KDtree_nn_interp(data_in, x_in, y_in, x_out, y_out, nnearest = 15, maxdist = None):
    """
    Nearest neighbour interpolation using scipy KDTree
    data_in: ndarray of float with shape (n1, n2)
        Data values to interpolate in input coordinate space
    x_in: ndarray of float with shape (n1, n2)
        x values of input coordinate space (e.g., require conversion from polar to Catesian first)
    y_in: ndarray of float with shape (n1, n2)
        y values of input coordinate space
    x_out: ndarray of float with shape (n1a, n2a)
        x values of output coordinate space
    y_out: ndarray of float with shape (n1a, n2a)
        x values of output coordinate space
    nnearest: int
        maximum number of nearest neighbours to consider when filling NaN values
    maxdist: float (in units of Cartesian space)
        maximum distance of nearest neighbours to consider when filling NaN values
        
        
    Returns: ndarray of float with shape (n1a, n2a)
    """
    
    def _make_coord_arrays(x):
        """
        Make sure that the coordinates are provided as ndarray
        of shape (numpoints, ndim)
        Parameters
        ----------
        x : ndarray of float with shape (numpoints, ndim)
            OR a sequence of ndarrays of float with len(sequence)==ndim and
            the length of the ndarray corresponding to the number of points
        """
        if type(x) in [list, tuple]:
            x = [item.ravel() for item in x]
            x = np.array(x).transpose()
        elif type(x) == np.ndarray:
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            elif x.ndim == 2:
                pass
            else:
                raise Exception("Cannot deal wih 3-d arrays, yet.")
        return x

    #transform output coordinates into pairs of coordiantes
    coord_out = _make_coord_arrays([x_out.ravel(), y_out.ravel()])
    vals_in = data_in.ravel()

    #build KDTree
    tree = cKDTree(np.c_[x_in.ravel(), y_in.ravel()])

    #query tree using output coordinates
    dists, idx = tree.query(coord_out, k=nnearest+1)
    # avoid bug, if there is only one neighbor at all
    if dists.ndim == 1:
        dists = dists[:, np.newaxis]
        idx = idx[:, np.newaxis]
    # get first neighbour

    vals_out = vals_in[idx[:, 0]]
    dists_cp = dists[..., 0].copy()

    # iteratively fill NaN with next neighbours
    isnan = np.isnan(vals_out)
    nanidx = np.argwhere(isnan)[..., 0]
    if nnearest > 1 & np.count_nonzero(isnan):
        for i in range(nnearest - 1):
            vals_out[isnan] = vals_in[idx[:, i + 1]][isnan]
            dists_cp[nanidx] = dists[..., i + 1][nanidx]
            isnan = np.isnan(vals_out)
            nanidx = np.argwhere(isnan)[..., 0]
            if not np.count_nonzero(isnan):
                break

    #apply max distance
    if maxdist is not None:
        vals_out = np.where(dists_cp > maxdist, np.nan, vals_out)

    return np.reshape(vals_out, x_out.shape)