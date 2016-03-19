# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""
Aggregator for a set of regions.

"""

from algorithm.unsed.neighbor import *


class Aggregator(object):
    """
    Aggregator for a set of regions.
    """

    def __init__(self, agg_type='UWA'):
        """
        Parameters
        aggr_type: str, optional
            Description for the method to  aggregate the regions. Supported methods include
            'uniform weighted average'(UWA), 'magnitude weighted average'(MWA), 'homogeneity
             weighted _average'(HWA), default is UWA.
        """
        if agg_type == 'UWA' or agg_type == 'MWA' or agg_type == 'HWA':
            self.agg_type = agg_type
        else:
            raise ValueError("The Type of aggregator should be 'UWA', 'MWA', and 'HWA'.")

    def compute(self, region, image):
        """
        Aggregate a set of regions
        Parameters
        region: A list of list of region objects to be aggregated..
            A 2d lists. the first dimension is for seeds, the second dimension is for threshold
        image: numpy 2d/3d/4d/ array
            image to be  segmented.
        Returns
        agg_image: numpy 2d/3d array
            Final segmentation from aggregating a set of regions
        """
        if image.ndim == 2:
            agg_image = np.zeros((image.shape[0], image.shape[1], len(region[0])), dtype=float)
            region_image = np.zeros((image.shape[0], image.shape[1], len(region)), dtype=int)
            weight = np.ones(len(region[0]))

            if self.agg_type == 'UWA':
                for i in range(len(region)):
                    for j in range(len(region[0])):
                        label = region[i][j].get_label()
                        region_image[label[:, 0], label[:, 1], j] = 1

                    weight = weight / weight.sum()
                    agg_image[:, :, i] = np.average(region_image, axis=2, weights=weight)
            elif self.agg_type == 'MWA':
                for i in range(len(region)):
                    for j in range(len(region[0])):
                        label = region[j][i].get_label()
                        region_image[label[:, 0], label[:, 1], j] = 1
                        weight[j] = np.mean(image[label[:, 0], label[:, 1]])

                    weight = weight / weight.sum()
                    agg_image[:, :, i] = np.average(region_image, axis=2, weights=weight)
            elif self.agg_type == 'HWA':
                for i in range(len(region)):
                    for j in range(len(region[0])):
                        label = region[j][i].get_label()
                        region_image[label[:, 0], label[:, 1], j] = 1
                        weight[j] = np.std(image[label[:, 0], label[:, 1]])

                    weight = weight / weight.sum()
                    agg_image[:, :, i] = np.average(region_image, axis=2, weights=weight)
        elif image.ndim == 3:
            agg_image = np.zeros((image.shape[0], image.shape[1], image.shape[2], len(region[0])), dtype=float)
            region_image = np.zeros((image.shape[0], image.shape[1], image.shape[2], len(region)), dtype=np.int32)
            weight = np.ones(len(region), dtype=float)

            if self.agg_type == 'UWA':
                for i in range(len(region[0])):
                    for j in range(region):
                        label = region[j][i].get_label()
                        region_image[label[:, 0], label[:, 1], label[:, 2], j] = 1

                    weight = weight / weight.sum()
                    agg_image[:, :, :, i] = np.average(region_image, axis=3, weights=weight)
            elif self.agg_type == 'MWA':
                for i in range(len(region[0])):
                    for j in range(len(region)):
                        label = region[j][i].get_label()
                        region_image[label[:, 0], label[:, 1], label[:, 2], j] = 1
                        weight[j] = np.mean(image[label[:, 0], label[:, 1], label[:, 2]])

                    if weight.min() < 0:
                        weight += np.abs(weight.min())
                    weight = weight / weight.sum()
                    agg_image[..., i] = np.average(region_image, axis=3, weights=weight)
            elif self.agg_type == 'HWA':
                for i in range(len(region[0])):
                    for j in range(len(region)):
                        label = region[j][i].get_label()
                        region_image[label[:, 0], label[:, 1], label[:, 2], j] = 1
                        weight[j] = np.std(image[label[:, 0], label[:, 1], label[:, 2]])

                    if weight.min() < 0:
                        weight = weight + np.abs(weight.min())
                    weight = weight / weight.sum()
                    agg_image[:, :, :, i] = np.average(region_image, axis=3, weights=weight)
        elif image.ndim == 4:
            pass

        return  agg_image