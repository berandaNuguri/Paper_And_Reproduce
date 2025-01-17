import numpy as np
from .firstNonSingletonDim import first_non_singleton_dim

def circshift_main(a, p, dim=None):
    """
    Circularly shift elements in the array.
    
    Parameters:
    a : array_like
        Input array.
    p : int or sequence of int
        The number of positions by which elements are shifted.
    dim : int, optional
        The dimension along which to shift. If not specified, shifts along the first non-singleton dimension.
    
    Returns:
    b : ndarray
        The shifted array.
    """
    a = np.asarray(a)
    
    if dim is not None:
        if not (np.isscalar(p) and np.isscalar(dim)):
            raise ValueError("dim must be a scalar")
        if not (isinstance(dim, int) and dim > 0):
            raise ValueError("dim must be a positive integer")
    elif np.isscalar(p):
        dim = first_non_singleton_dim(a)
    else:
        dim = 0
    
    if not (np.isscalar(p) or (isinstance(p, (list, tuple, np.ndarray)) and all(isinstance(x, int) for x in p))):
        raise ValueError("p must be an integer or a sequence of integers")
    
    if np.isscalar(p):
        p = [p]
    
    numDimsA = a.ndim
    if dim > 1:
        p = [0] * (dim - 1) + p
    
    if len(p) < numDimsA:
        p = list(p) + [0] * (numDimsA - len(p))
    
    # Initialize the list of indices
    idx = [slice(None)] * numDimsA
    
    # Calculate the shifted indices
    for k in range(numDimsA):
        if p[k] != 0:
            m = a.shape[k]
            idx[k] = np.mod(np.arange(m) - p[k] % m, m)
    
    b = a[tuple(idx)]
    
    return b

def circshift(a, p, dim=None):
    """
    Wrapper function for circshift to handle exceptions.
    
    Parameters:
    a : array_like
        Input array.
    p : int or sequence of int
        The number of positions by which elements are shifted.
    dim : int, optional
        The dimension along which to shift. If not specified, shifts along the first non-singleton dimension.
    
    Returns:
    b : ndarray
        The shifted array.
    """
    try:
        if dim is None:
            b = circshift_main(a, p)
        else:
            b = circshift_main(a, p, dim)
    except Exception as e:
        raise e
    
    return b