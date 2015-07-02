# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""
Optimizer to select the optimal segmentation from a set of region growing results.

"""

from algorithm.unsed.neighbor import *

class Optimizer(object):
    """
    Optimizer to select the optimal segmentation from a set of region growing results.
    Attributes
    opt_type:  str, optional
        Description for the criteria to select the optimal segmentation from region growing. methods include
        'peripheral contrast'(PC), 'average contrast'(AC), 'homogeneity  weighted _average'(HWA),
        default is PC.

    """
    def __init__(self, opt_type):
        """
        Parameters
        opt_type:  str, optional
            Description for the criteria to select the optimal segmentation from region growing. methods include
            'peripheral contrast'(PC), 'average contrast'(AC), 'homogeneity  weighted _average'(HWA),
            default is PC.
        """
        if opt_type == 'PC' or opt_type == 'AC':
            self.opt_type = opt_type
        else:
            raise ValueError("The Type of aggregator should be 'PC' and 'AC'.")
        self.opt_type = opt_type

    def compute(self, region, image):
        """
        Find the optimal segmentation according to the specified optimization criteria
        Parameters
        region: A 2d list of region object
            A 2d lists. the first dimension is for seeds, the second dimension is for threshold
        image: numpy 2d/3d/4d/ array
            image to be  segmented.
        Returns
        con_val: optimizing index
            A index to measure the performance of the segmentation
        """
        con_val = np.zeros((len(region), len(region[0])))
        if image.ndim == 2:
            if self.opt_type == 'PC':
                for i in range(len(region)):
                    for j in range(len(region[0])):
                        bound = region[i][j].compute_boundary()
                        neighbor = region[i][j].get_neighbor()
                        bound_val = np.mean(image[bound[:, 0], bound[:, 1]])
                        per_val = np.mean(image[neighbor[:, 0], neighbor[:, 1]])
                        con_val[i, j] = (bound_val - per_val) / (bound_val + per_val)
            elif self.opt_type == 'AC':
                for i in range(len(region)):
                    for j in range(len(region[0])):
                        label = region[i][j].get_label()
                        neighbor = region[i][j].get_neighbor()
                        region_val = np.mean(image[label[:, 0], label[:, 1]])
                        per_val = np.mean(image[neighbor[:, 0], neighbor[:, 1]])
                        con_val[i, j] = (region_val - per_val) / (region_val + per_val)
        elif image.ndim == 3:
            if self.opt_type == 'PC':
                for i in range(len(region)):
                    for j in range(len(region[0])):
                        bound = region[i][j].compute_boundary()
                        neighbor = region[i][j].get_neighbor()
                        bound_val = np.mean(image[bound[:, 0], bound[:, 1], bound[:, 2]])
                        per_val = np.mean(image[neighbor[:, 0], neighbor[:, 1], neighbor[:, 2]])
                        if (bound_val + per_val) == 0:
                            con_val[i, j] = 0
                        else:
                            con_val[i, j] = (bound_val - per_val) / (bound_val + per_val)
            elif self.opt_type == 'AC':
                for i in range(len(region)):
                    for j in range(len(region[0])):
                        label = region[i][j].get_label()
                        neighbor = region[i][j].get_neighbor()
                        region_val = np.mean(image[label[:, 0], label[:, 1], label[:, 2]])
                        per_val = np.mean(image[neighbor[:, 0], neighbor[:, 1], neighbor[:, 2]])
                        con_val[i, j] = (region_val - per_val)
        elif image.ndim == 4:
            pass
        con_val = con_val / con_val.max()

        return con_val