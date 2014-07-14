# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""
Neighbor Generator for a set of refer points(pixels or voxels).

"""
import numpy as np


class SpatialNeighbor(object):
    """Define spatial neighbor for a region which consists of of points(pixels or voxels).
    
    Attributes
    ----------
    offsets: 2d numpy array(Nx2 or Nx3)
        Each row represents a relative offset to the ref points
    image_dims: int
        The spatial dimension of the target image(e.g.,2 or 3)
    neighbor_size: int
        The size of neighbor for a points(e.g. 26)

    """

    def __init__(self, image_dims, neighbor_size):
        """

        Parameters
        ----------
        image_dims: int
            The spatial dimension of the target image(e.g.,2 or 3)
        neighbor_size: int
            The size of neighbor for a points(e.g. 26)

        """

        if len(image_dims) == 2 or len(image_dims) == 3:
            self.image_dims = image_dims
            self.neighbor_size = neighbor_size

        else:
            raise ValueError("The image dimension should be 2 or 3")


    def compute(self, ref):
        return self.computing(ref)


class Connectivity(SpatialNeighbor):
    """

    Define neighbors which are at least connected with a ref point.

    """

    def __init__(self, image_dims, neighbor_size):

        """

        Parameters
        ----------
        image_dims: int
            The spatial dimension of the target image(e.g.,2 or 3)
        neighbor_size: int
            The size of neighbor for a points(e.g. 26)
        """

        super(Connectivity, self).__init__(image_dims, neighbor_size)

        offsets = []
        if len(self.image_dims) == 2:  # 2D image 4, 6, 8-connected
            if self.neighbor_size == 4:
                offsets = np.array([[1, 0], [-1, 0],
                                    [0, 1], [0, -1]])
            elif self.neighbor_size == 6:
                offsets = np.array([[1, 0], [-1, 0],
                                    [0, 1], [0, -1],
                                    [1, 1], [-1, -1]])
            elif self.neighbor_size == 8:
                offsets = np.array([[1, 0], [-1, 0],
                                    [0, 1], [0, -1],
                                    [1, 1], [-1, -1],
                                    [1, -1], [-1, 1]])
        elif len(self.image_dims) == 3:  # 3D volume 6, 18, 26-connected
            if self.neighbor_size == 6:
                offsets = np.array([[1, 0, 0], [-1, 0, 0],
                                    [0, 1, 0], [0, -1, 0],
                                    [0, 0, -1], [0, 0, -1]])
            elif self.neighbor_size == 18:
                offsets = np.array([[0, -1, -1], [-1, 0, -1], [0, 0, -1],
                                    [1, 0, -1], [0, 1, -1], [-1, -1, 0],
                                    [0, -1, 0], [1, -1, 0], [-1, 0, 0],
                                    [1, 0, 0], [-1, 1, 0], [0, 1, 0],
                                    [1, 1, 0], [0, -1, 1], [-1, 0, 1],
                                    [0, 0, 1], [1, 0, 1], [0, 1, 1]])

            elif self.neighbor_size == 26:
                offsets = np.array([[-1, -1, -1], [0, -1, -1], [1, -1, -1],
                                    [-1, 0, -1], [0, 0, -1], [1, 0, -1],
                                    [-1, 1, -1], [0, 1, -1], [1, 1, -1],
                                    [-1, -1, 0], [0, -1, 0], [1, -1, 0],
                                    [-1, 0, 0], [1, 0, 0], [-1, 1, 0],
                                    [0, 1, 0], [1, 1, 0], [-1, -1, 1],
                                    [0, -1, 1], [1, -1, 1], [-1, 0, 1],
                                    [0, 0, 1], [1, 0, 1], [-1, 1, 1],
                                    [0, 1, 1], [1, 1, 1]])

        self.offsets = offsets

    def computing(self, ref):
        """

        Parameters
        ----------
        refs list or nd array
            Each row represents the coordinates for a ref points
        """

        ref = np.array(ref)

        coors = ref + self.offsets
        return coors[is_in_image(coors, self.image_dims), :]


class Sphere(SpatialNeighbor):
    """

    Define sphere neighbor for a pixel

    """

    def __init__(self, image_dims, neighbor_size):

        super(Sphere, self).__init__(image_dims, neighbor_size)

        offsets = []
        if self.image_dims == 2:
            for x in np.arange(-self.neighbor_size, self.neighbor_size + 1):
                for y in np.arange(-self.neighbor_size, self.neighbor_size + 1):
                    if np.linalg.norm([x, y]) <= self.neighbor_size:
                        offsets.append([x, y])

        elif self.image_dims == 3:
            for x in np.arange(-self.neighbor_size, self.neighbor_size + 1):
                for y in np.arange(-self.neighbor_size, self.neighbor_size + 1):
                    for z in np.arange(-self.neighbor_size, self.neighbor_size + 1):
                        if np.linalg.norm([x, y, z]) <= self.neighbor_size:
                            offsets.append([x, y, z])

        self.offsets = offsets

    def computing(self, ref):
        coors = ref + np.array(self.offsets)
        return coors[is_in_image(coors, self.image_dims), :]


class Cube(SpatialNeighbor):
    """
    Define cube neighbor for a pixel

    """

    def __init__(self, image_dims, neighbor_size):
        super(Cube, self).__init__(image_dims, neighbor_size)

        offsets = []
        if self.image_dims == 2:
            for x in np.arange(-self.neighbor_size, self.neighbor_size + 1):
                for y in np.arange(-self.neighbor_size, self.neighbor_size + 1):
                    offsets.append([x, y])

        elif self.image_dims == 3:
            for x in np.arange(-self.neighbor_size, self.neighbor_size + 1):
                for y in np.arange(-self.neighbor_size, self.neighbor_size + 1):
                    for z in np.arange(-self.neighbor_size, self.neighbor_size + 1):
                        offsets.append([x, y, z])

        self.offsets = offsets

    def computing(self, ref):
        coors = ref + np.array(self.offsets)
        return coors[is_in_image(coors, self.image_dims), :]


def is_in_image(coors, image_shape):
    """
    Check whether the coordinates is in the range of image.

    Parameters
    ----------
    coors: 2d/3d numpy array
    image_shape: a numpy array represent the shape of image
    """
    if not isinstance(coors, np.ndarray):
        coors = np.array(coors)
    if len(coors.shape) == 1:
        coors = coors.reshape(1, len(image_shape))

    if len(coors.shape) == 2:
        return np.all([coors[:, 0] >= 0, coors[:, 0] < image_shape[0],
                       coors[:, 1] >= 0, coors[:, 1] < image_shape[1]], axis=0)
    elif len(coors.shape) == 3:
        return np.all([coors[:, 0] >= 0, coors[:, 0] < image_shape[0],
                       coors[:, 1] >= 0, coors[:, 1] < image_shape[1],
                       coors[:, 2] >= 0, coors[:, 2] < image_shape[2]], axis=0)


if __name__ == "__main__":
    conn = Connectivity((20, 20, 20), 26)
    print 'Connectivity\n', conn.compute((20, 15, 15))

    sph = Sphere((20, 20, 20), 3)
    print 'Sphere\n', sph.compute((20, 19, 18))

    cube = Cube((20, 20, 20), 3)
    print 'Cube\n', cube.compute((20, 19, 18))







