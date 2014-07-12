import numpy as np


def is_inside(coors, image_shape):
    """
    whether the coordinate in the range of image.
    """
    if not isinstance(coors, np.ndarray):
        coors = np.array(coors)
    if len(coors.shape) == 1:
        coors = coors.reshape(1, len(image_shape))

    if len(coors.shape) == 2:
        return  np.all([coors[:, 0] >= 0, coors[:, 0] < image_shape[0],
                        coors[:, 1] >= 0, coors[:, 1] < image_shape[1]], axis=0)
    elif len(coors.shape) == 3:
        return  np.all([coors[:, 0] >= 0, coors[:, 0] < image_shape[0],
                        coors[:, 1] >= 0, coors[:, 1] < image_shape[1],
                        coors[:, 2] >= 0, coors[:, 2] < image_shape[2]], axis=0)


def intersect2d(A, B):
    """
    A,B: 2D numpy array
    """
    nrows, ncols = A.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)], 'formats': ncols * [A.dtype]}

    C = np.intersect1d(A.view(dtype), B.view(dtype))
    C = C.view(A.dtype).reshape(-1, ncols)

    return C


def in2d(A, B):
    """
    A,B: 2D numpy array
    Return a boolean array the same length as A that is true where an element of A is in B and False otherwise
    """
    nrows, ncols = A.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)], 'formats': ncols * [A.dtype]}

    C = np.in1d(A.view(dtype), B.view(dtype))

    return C

