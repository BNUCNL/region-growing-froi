import numpy as np


def intersect2d(A, B):
    """
    Find the same rows of A and B

    Parameters
    ----------
    A,B: numpy 2d array

    Returns
    -------
    C: numpy 2d array
        A array with rows that are in both A and B
    """
    nrows, ncols = A.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)], 'formats': ncols * [A.dtype]}

    C = np.intersect1d(A.view(dtype), B.view(dtype))
    C = C.view(A.dtype).reshape(-1, ncols)

    return C


def in2d(A, B):
    """
    Find whether the row of A is in B

    Parametersasrgtest.py
    ----------
    A,B: numpy 2d array

    Returns
    -------
    C: 1d array
        A boolean array with the same length as A that is true where an element of A is in B and False otherwise
    """
    nrows, ncols = A.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)], 'formats': ncols * [A.dtype]}

    C = np.in1d(np.ascontiguousarray(A).view(dtype), np.ascontiguousarray(B).view(dtype))

    return C


def unique2d(A):
    """
    find the unique row of a 2d numpy array

    Parameters
    ----------
    A: 2D numpy array

    Return
    ------
    UA: the array only unique rows are left
    """

    B = np.ascontiguousarray(A).view(np.dtype((np.void, A.dtype.itemsize * A.shape[1])))

    UA = np.unique(B).view(A.dtype).reshape(-1, A.shape[1])

    return UA

