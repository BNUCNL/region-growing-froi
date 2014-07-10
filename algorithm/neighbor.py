# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""
Generate neighbor for a refer point.

"""
import numpy as np


class SpatialNeighbor(object):
    """Define spatial neighbor for a pixel or voxel.
    
    Return
    ------
        offsets: 2xN or 3xN np array
    
    Contributions
    -------------
        Author: Zonglei Zhen
        Editor: 
    
    """

    def __init__(self, img_shape, nb_size):
        if len(img_shape) == 2 or len(img_shape) == 3:
            self.img_shape = img_shape
            self.nb_size = nb_size

        else:
            raise ValueError("The image dimension should be 2 or 3")


    def compute(self, ref):
        return self.computing(ref)


class Connectivity(SpatialNeighbor):
    """Define pixel connectivity for 2D or 3D image.

    Returns
    -------
        offsets: 2 x N or 3 x N np array
    
    Contributions
    -------------
        Author: Zonglei Zhen
        Editor: 
    
    """

    def __init__(self, img_shape, nb_size):

        """

        :rtype : object
        """
        super(Connectivity, self).__init__(img_shape, nb_size)

        offsets = []
        if len(self.img_shape) == 2:  # 2D image 4, 6, 8-connected
            if self.nb_size == 4:
                offsets = np.array([[1, 0], [-1, 0],
                                    [0, 1], [0, -1]])
            elif self.nb_size == 6:
                offsets = np.array([[1, 0], [-1, 0],
                                    [0, 1], [0, -1],
                                    [1, 1], [-1, -1]])
            elif self.nb_size == 8:
                offsets = np.array([[1, 0], [-1, 0],
                                    [0, 1], [0, -1],
                                    [1, 1], [-1, -1],
                                    [1, -1], [-1, 1]])
        elif len(self.img_shape) == 3:  # 3D volume 6, 18, 26-connected
            if self.nb_size == 6:
                offsets = np.array([[1, 0, 0], [-1, 0, 0],
                                    [0, 1, 0], [0, -1, 0],
                                    [0, 0, -1], [0, 0, -1]])
            elif self.nb_size == 18:
                offsets = np.array([[0, -1, -1], [-1, 0, -1], [0, 0, -1],
                                    [1, 0, -1], [0, 1, -1], [-1, -1, 0],
                                    [0, -1, 0], [1, -1, 0], [-1, 0, 0],
                                    [1, 0, 0], [-1, 1, 0], [0, 1, 0],
                                    [1, 1, 0], [0, -1, 1], [-1, 0, 1],
                                    [0, 0, 1], [1, 0, 1], [0, 1, 1]])

            elif self.nb_size == 26:
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
        coors = ref + self.offsets
        return coors[is_in_image(coors, self.img_shape), :]


class Sphere(SpatialNeighbor):
    """Sphere neighbor for pixels or voxels.
    
    Contributions
    -------------
        Author: Zonglei Zhen
        Editor: 
    
    """

    def __init__(self, img_shape, nb_size):

        super(Sphere, self).__init__(img_shape, nb_size)

        offsets = []
        if len(self.img_shape) == 2:
            for x in np.arange(-self.nb_size, self.nb_size + 1):
                for y in np.arange(-self.nb_size, self.nb_size + 1):
                    if np.linalg.norm([x, y]) <= self.nb_size:
                        offsets.append([x, y])

        elif len(self.img_shape) == 3:
            for x in np.arange(-self.nb_size, self.nb_size + 1):
                for y in np.arange(-self.nb_size, self.nb_size + 1):
                    for z in np.arange(-self.nb_size, self.nb_size + 1):
                        if np.linalg.norm([x, y, z]) <= self.nb_size:
                            offsets.append([x, y, z])

        self.offsets = offsets

    def computing(self, ref):
        coors = ref + np.array(self.offsets)
        return coors[is_in_image(coors, self.img_shape), :]


class Cube(SpatialNeighbor):
    """
    
    Contributions
    -------------
        Author: Zonglei Zhen
        Editor: 
    
    """

    def __init__(self, img_shape, nb_size):
        super(Cube, self).__init__(img_shape, nb_size)

        offsets = []
        if len(self.img_shape) == 2:
            for x in np.arange(-self.nb_size, self.nb_size + 1):
                for y in np.arange(-self.nb_size, self.nb_size + 1):
                    offsets.append([x, y])

        elif len(self.img_shape) == 3:
            for x in np.arange(-self.nb_size, self.nb_size + 1):
                for y in np.arange(-self.nb_size, self.nb_size + 1):
                    for z in np.arange(-self.nb_size, self.nb_size + 1):
                        offsets.append([x, y, z])

        self.offsets = offsets

    def computing(self, ref):
        coors = ref + np.array(self.offsets)
        return coors[is_in_image(coors, self.img_shape), :]


def is_in_image(coors, image_shape):
    """
    check whether the coordinates is in the range of image.
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







