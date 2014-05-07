
import numpy as np
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
        else:
            raise ValueError("2D data must be 4/6/8 neighbors.")
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
    else:
        raise ValueError("Input data dimension error!.")

    return offsets
