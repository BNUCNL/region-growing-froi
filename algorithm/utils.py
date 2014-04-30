
import numpy as np

def inside(coors, image_shape):
    """
    whether the coordinate in the range of image.
    """
    if not isinstance(coors, np.ndarray):
        coors = np.array(coors)
    if len(coors.shape) == 1:
        coors = coors.reshape(1, 3)
    return  np.all([coors[:, 0] >= 0, coors[:, 0] < image_shape[0],
                    coors[:, 1] >= 0, coors[:, 1] < image_shape[1],
                    coors[:, 2] >= 0, coors[:, 2] < image_shape[2]], axis=0)

def compute_offsets(nbdim, nbsize):
    """
    compute the offsets.
    """
    if nbdim == 2:
        # 2D image 4, 6, 8-connected
        if nbsize == 4:
            offsets = np.array([[1, 0],
                                [-1, 0],
                                [0, 1],
                                [0, -1]])
        elif nbsize == 6:
            offsets = np.array([[1, 0],
                                [-1, 0],
                                [0, 1],
                                [0, -1],
                                [1, 1],
                                [-1, -1]])
        elif nbsize == 8:
            offsets = np.array([[1, 0],
                                [-1, 0],
                                [0, 1],
                                [0, -1],
                                [1, 1],
                                [-1, -1],
                                [1, -1],
                                [-1, 1]])
        else: raise ValueError("2D data must be 4/6/8 neighbors.")
    elif nbdim == 3:
        # 3D volume 6, 18, 26-connected
        if nbsize == 6:
            offsets = np.array([[1, 0, 0],
                                [-1, 0, 0],
                                [0, 1, 0],
                                [0, -1, 0],
                                [0, 0, -1],
                                [0, 0, -1]])
        elif nbsize == 18:
            offsets = np.array([[0, -1, -1],
                                [-1, 0, -1],
                                [0, 0, -1],
                                [1, 0, -1],
                                [0, 1, -1],
                                [-1, -1, 0],
                                [0, -1, 0],
                                [1, -1, 0],
                                [-1, 0, 0],
                                [1, 0, 0],
                                [-1, 1, 0],
                                [0, 1, 0],
                                [1, 1, 0],
                                [0, -1, 1],
                                [-1, 0, 1],
                                [0, 0, 1],
                                [1, 0, 1],
                                [0, 1, 1]])

        elif nbsize == 26:
            offsets = np.array([[-1, -1, -1],
                                [0, -1, -1],
                                [1, -1, -1],
                                [-1, 0, -1],
                                [0, 0, -1],
                                [1, 0, -1],
                                [-1, 1, -1],
                                [0, 1, -1],
                                [1, 1, -1],
                                [-1, -1, 0],
                                [0, -1, 0],
                                [1, -1, 0],
                                [-1, 0, 0],
                                [1, 0, 0],
                                [-1, 1, 0],
                                [0, 1, 0],
                                [1, 1, 0],
                                [-1, -1, 1],
                                [0, -1, 1],
                                [1, -1, 1],
                                [-1, 0, 1],
                                [0, 0, 1],
                                [1, 0, 1],
                                [-1, 1, 1],
                                [0, 1, 1],
                                [1, 1, 1]])
        else:
            raise ValueError("3D data must be 6/18/26 neighbors.")
    elif nbdim == 4:
        # 4D volume 26-connected
        if nbsize == 26:
            offsets = np.array([[-1, -1, -1],
                                [0, -1, -1],
                                [1, -1, -1],
                                [-1, 0, -1],
                                [0, 0, -1],
                                [1, 0, -1],
                                [-1, 1, -1],
                                [0, 1, -1],
                                [1, 1, -1],
                                [-1, -1, 0],
                                [0, -1, 0],
                                [1, -1, 0],
                                [-1, 0, 0],
                                [1, 0, 0],
                                [-1, 1, 0],
                                [0, 1, 0],
                                [1, 1, 0],
                                [-1, -1, 1],
                                [0, -1, 1],
                                [1, -1, 1],
                                [-1, 0, 1],
                                [0, 0, 1],
                                [1, 0, 1],
                                [-1, 1, 1],
                                [0, 1, 1],
                                [1, 1, 1]])
        else:
            raise ValueError("4D data must be 26 neighbors.")
    return offsets
