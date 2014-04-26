
import numpy as np

def inside(coors,image_shape):
    """
    whether the coordinate in the range of image.
    """
    if not isinstance(coors,np.ndarray):
        coors = np.array(coors)
    if len(coors.shape) == 1:
        coors = coors.reshape(1,3)
    return  np.all([coors[:,0] >= 0, coors[:,0] < image_shape[0], \
                    coors[:,1] >= 0, coors[:,1] < image_shape[1], \
                    coors[:,2] >= 0, coors[:,2] < image_shape[2]],axis = 0)
