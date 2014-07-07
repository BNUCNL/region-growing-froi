# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""
Generate neighbor for a refer point.

"""

import numpy as np



class SpatialNeighbor:
    """Define spatial neighbor for a pixel or voxel.
    
    Return
    ------
        offsets: 2xN or 3xN np array
    
    Contributions
    -------------
        Author: Zonglei Zhen
        Editor: 
    
    """
    def __init__(self, img_dim, nb_size, ref_points):

        if img_dim != 2 or img_dim != 3:
            raise ValueError("The image dimension should be 2 or 3")

        self.img_dim =  img_dim
        self.img_size  = nb_size
        self.ref_points = ref_points

    def compute(self):
        return self.computing()




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

    def __init__(self, img_dim, nb_size, ref_points):

        super(Connectivity,self).__init__(img_dim, nb_size, ref_points)

        if self.img_dim == 2: # 2D image 4, 6, 8-connected
            if self.nb_size == 4:
                offsets = np.array([[1, 0],[-1, 0],
                                    [0, 1],[0, -1]])
            elif self.nb_size == 6:
                offsets = np.array([[1, 0],[-1, 0],
                                    [0, 1],[0, -1],
                                    [1, 1], [-1, -1]])
            elif self.nb_size == 8:
                offsets = np.array([[1, 0],[-1, 0],
                                    [0, 1],[0, -1],
                                    [1, 1], [-1, -1]
                                    [1, -1], [-1, 1]])
        elif self.img_dim == 3: # 3D volume 6, 18, 26-connected
            if self.nb_size == 6:
                offsets = np.array([[1, 0, 0],[-1, 0, 0],
                                    [0, 1, 0],[0, -1, 0],
                                    [0, 0, -1], [0, 0, -1]])      
            elif self.nb_size == 18:
                offsets = np.array([[0,-1,-1],[-1, 0,-1],[0, 0,-1],
                                    [1, 0,-1],[0, 1,-1],[-1,-1, 0],
                                    [0,-1, 0],[1,-1, 0],[-1, 0, 0],
                                    [1, 0, 0],[-1, 1, 0],[0, 1, 0],
                                    [1, 1, 0],[0,-1, 1],[-1, 0, 1],
                                    [0, 0, 1],[1, 0, 1],[0, 1, 1]])
        
            elif self.nb_size == 26:
                offsets = np.array([[-1,-1,-1],[0,-1,-1],[1,-1,-1],
                                    [-1, 0,-1],[0, 0,-1],[1, 0,-1],
                                    [-1, 1,-1],[0, 1,-1],[1, 1,-1],
                                    [-1,-1, 0],[0,-1, 0],[1,-1, 0], 
                                    [-1, 0, 0],[1, 0, 0],[-1, 1, 0],
                                    [0, 1, 0],[1, 1, 0],[-1,-1, 1],
                                    [0,-1, 1],[1,-1, 1],[-1, 0, 1],
                                    [0, 0, 1],[1, 0, 1],[-1, 1, 1],
                                    [0, 1, 1],[1, 1, 1]])
        self.offsets = offsets.T


    def computing(self):
        coors = self.ref_points + self.offsets
        return coors(is_in_image(coors,self.img_size))


class Sphere(SpatialNeighbor):
    """Sphere neighbor for pixels or voxels.
    
    Contributions
    -------------
        Author: Zonglei Zhen
        Editor: 
    
    """


    def __init__(self, img_dim, nb_size, ref_points,):

        super(Sphere,self).__init__(img_dim, nb_size, ref_points)

        offsets = []
        if self.img_dim == 2:
            for x in np.arange(-self.nb_size, self.nb_size + 1):
                for y in np.arange(-self.nb_size, self.nb_size + 1):
                    if np.linalg.norm([x,y]) <= self.nb_size:
                            offsets.append([x,y])

        elif self.img_dim == 3:
            for x in np.arange(-self.nb_size, self.nb_size + 1):
                for y in np.arange(-self.nb_size, self.nb_size + 1):
                    for z in np.arange(-self.nb_size, self.nb_size + 1):
                        if np.linalg.norm([x,y,z]) <= self.nb_size:
                            offsets.append([x,y,z])

        self.offsets = offsets



    def computing(self):
        coors = self.ref_points + self.offsets
        return coors(is_in_image(coors,self.img_size))

     
class Cube(SpatialNeighbor):
    """
    
    Contributions
    -------------
        Author: Zonglei Zhen
        Editor: 
    
    """

    def __init__(self, img_dim, nb_size, ref_points,):
        super(Cube,self).__init__(img_dim, nb_size, ref_points)

        offsets = []
        if self.nb_dim == 2:
            for x in np.arange(-self.nb_size, self.nb_size + 1):
                for y in np.arange(-self.nb_size, self.nb_size + 1):
                    offsets.append([x,y])

        elif self.nb_dim == 3:
            for x in np.arange(-self.nb_size, self.nb_size + 1):
                for y in np.arange(-self.nb_size, self.nb_size + 1):
                    for z in np.arange(-self.nb_size, self.nb_size + 1):
                        offsets.append([x,y,z])


    def computing(self):
        coors = self.ref_points + self.offsets
        return coors(is_in_image(coors,self.img_size))




def is_in_image(coors, image_shape):
    """
    check whether the coordinates is in the range of image.
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



