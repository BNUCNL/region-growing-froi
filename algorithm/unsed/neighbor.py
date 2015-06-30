# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""
SpatialNeighbor Generator for a set of refer points(pixels or voxels).

"""
import numpy as np
from algorithm.unsed import utils


class SpatialNeighbor(object):
    """Define spatial neighbor for a region which consists of of points(pixels or voxels).
    
    Attributes
    ----------
    offsets: 2d numpy array(Nx2 or Nx3)
        Each row represents a relative offset to the ref points
    neighbor_type: str
        Type for the neighbors. Supported types include 'connected', 'sphere', and 'cube'
    neighbor_size: int
        The size of neighbor for a points(e.g. 26)
     image_shape: tuple or list
        The spatial shape of the target image(e.g.,(64,64,32))
    """

    def __init__(self, neighbor_type, image_shape, neighbor_size):
        """

        Parameters
        ----------
        neighbor_type: str
            Type for the neighbors. Supported types include 'connected', 'sphere', and 'cube'
        neighbor_size: int
            The size of neighbor for a points(e.g. 26)
        image_shape: int
            The spatial shape of the target image(e.g.,2 or 3)
        """

        if len(image_shape) == 2 or len(image_shape) == 3:
            self.image_shape = image_shape
            self.neighbor_size = neighbor_size

        else:
            raise ValueError("The spatial dimension of image should be 2 or 3")

        offsets = []
        if neighbor_type == 'connected':
            if len(self.image_shape) == 2:  # 2D image 4, 6, 8-connected
                if self.neighbor_size == 4:
                    offsets = [[1, 0], [-1, 0],
                               [0, 1], [0, -1]]
                elif self.neighbor_size == 6:
                    offsets = [[1, 0], [-1, 0],
                               [0, 1], [0, -1],
                               [1, 1], [-1, -1]]
                elif self.neighbor_size == 8:
                    offsets = [[1, 0], [-1, 0],
                               [0, 1], [0, -1],
                               [1, 1], [-1, -1],
                               [1, -1], [-1, 1]]
            elif len(self.image_shape) == 3:  # 3D volume 6, 18, 26-connected
                if self.neighbor_size == 6:
                    offsets = [[1, 0, 0], [-1, 0, 0],
                               [0, 1, 0], [0, -1, 0],
                               [0, 0, -1], [0, 0, -1]]
                elif self.neighbor_size == 18:
                    offsets = [[0, -1, -1], [-1, 0, -1], [0, 0, -1],
                               [1, 0, -1], [0, 1, -1], [-1, -1, 0],
                               [0, -1, 0], [1, -1, 0], [-1, 0, 0],
                               [1, 0, 0], [-1, 1, 0], [0, 1, 0],
                               [1, 1, 0], [0, -1, 1], [-1, 0, 1],
                               [0, 0, 1], [1, 0, 1], [0, 1, 1]]

                elif self.neighbor_size == 26:
                    offsets = [[-1, -1, -1], [0, -1, -1], [1, -1, -1],
                               [-1, 0, -1], [0, 0, -1], [1, 0, -1],
                               [-1, 1, -1], [0, 1, -1], [1, 1, -1],
                               [-1, -1, 0], [0, -1, 0], [1, -1, 0],
                               [-1, 0, 0], [1, 0, 0], [-1, 1, 0],
                               [0, 1, 0], [1, 1, 0], [-1, -1, 1],
                               [0, -1, 1], [1, -1, 1], [-1, 0, 1],
                               [0, 0, 1], [1, 0, 1], [-1, 1, 1],
                               [0, 1, 1], [1, 1, 1]]
        elif neighbor_type == 'sphere':
            if len(self.image_shape) == 2:
                for x in np.arange(-self.neighbor_size, self.neighbor_size + 1):
                    for y in np.arange(-self.neighbor_size, self.neighbor_size + 1):
                        if np.linalg.norm([x, y]) <= self.neighbor_size:
                            offsets.append([x, y])

            elif len(self.image_shape) == 3:
                for x in np.arange(-self.neighbor_size, self.neighbor_size + 1):
                    for y in np.arange(-self.neighbor_size, self.neighbor_size + 1):
                        for z in np.arange(-self.neighbor_size, self.neighbor_size + 1):
                            if np.linalg.norm([x, y, z]) <= self.neighbor_size:
                                offsets.append([x, y, z])
        elif neighbor_type == 'cube':
            if len(self.image_shape) == 2:
                for x in np.arange(-self.neighbor_size, self.neighbor_size + 1):
                    for y in np.arange(-self.neighbor_size, self.neighbor_size + 1):
                        offsets.append([x, y])

            elif len(self.image_shape) == 3:
                for x in np.arange(-self.neighbor_size, self.neighbor_size + 1):
                    for y in np.arange(-self.neighbor_size, self.neighbor_size + 1):
                        for z in np.arange(-self.neighbor_size, self.neighbor_size + 1):
                            offsets.append([x, y, z])
        else:
            raise ValueError("The Neighbor Type should be 'connected', 'sphere', and 'cube'.")

        self.offsets = np.array(offsets)

    def compute(self, refs):
        """
        compute the neighbor for a region(i.e., a set of pixes or voxels)
        Parameters
        ----------
        refs: list, tuple or numpy 2d array
            Each row represents the coordinates for a ref point

        """
        if not isinstance(refs, np.ndarray):
            refs = np.array(refs)
        if len(refs.shape) != 2:
            refs = refs.reshape(-1, 3)

        coords = np.zeros((self.offsets.shape[0] * refs.shape[0], refs.shape[1]), dtype=int)
        for r in range(refs.shape[0]):
            coords[r * self.offsets.shape[0]:(r + 1) * self.offsets.shape[0], :] = refs[r, :] + self.offsets
        coords = coords[is_in_image(coords, self.image_shape), :]

        return utils.unique2d(coords)

def is_in_image(coords, image_shape):
    """
    Check whether the coordinates is in the range of image.

    Parameters
    ----------
    coors: 2d/3d numpy array
    image_shape: a numpy array represent the shape of image
    
    """
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)

    if coords.ndim == 2:
        return np.all([coords[:, 0] >= 0, coords[:, 0] < image_shape[0],
                       coords[:, 1] >= 0, coords[:, 1] < image_shape[1]], axis=0)
    elif coords.ndim == 3:
        return np.all([coords[:, 0] >= 0, coords[:, 0] < image_shape[0],
                       coords[:, 1] >= 0, coords[:, 1] < image_shape[1],
                       coords[:, 2] >= 0, coords[:, 2] < image_shape[2]], axis=0)






