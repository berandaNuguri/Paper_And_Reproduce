import numpy as np

def first_non_singleton_dim(x):
    """
    Return the index of the first non-singleton dimension.
    
    Parameters:
    x : array_like
        Input array.
    
    Returns:
    dim : int
        The index of the first non-singleton dimension. If x is a scalar, returns 1.
    """
    if np.isscalar(x):
        return 1
    
    x = np.asarray(x)
    for k in range(x.ndim):
        if x.shape[k] != 1:
            return k + 1  # MATLAB is 1-based, Python is 0-based, so add 1
    
    return 1  # If all dimensions are singleton, return 1