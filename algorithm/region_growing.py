# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""
Seeded region growing performs a segmentation of an image with respect to a set of points, known as seeded region.

"""
import copy
import numpy as np

from region import Region

class SeededRegionGrowing(object):
    def __init__(self, seed, stop_size = 300):
        self.seed = seed
        self.stop_size = stop_size

    def set_seed(self, seed):
        self.seed = seed

    def get_seed(self):
        return self.seed

    def convert_image_to_regions(self, image, mask=None):
        unique_image = np.arange(image.size).reshape(image.shape) + 1
        if mask is not None:
            unique_image[mask == False] = 0 #0 stands for the background
        value_region_dicts = {}
        values_len = (unique_image > 0).sum()
        unique_values = np.unique(unique_image)
        for i in range(1, unique_values.size):
            # print 'i: ', i, '   generate region...'
            value_region_dicts[unique_values[i]] = Region(unique_values[i], image, unique_image)

        return value_region_dicts, unique_image

    def compute_nearest_region(self, result_region, value_region_dicts):
        result_region.compute_neighbor_regions()
        neighbor_region_values = list(result_region.get_neighbor_region_value())

        print 'result_region.get_region_mean(): ', result_region.get_region_mean()
        theta = np.zeros((len(neighbor_region_values), ))
        for i in range(len(neighbor_region_values)):
            theta[i] = abs(value_region_dicts[neighbor_region_values[i]].get_region_mean()
                           - result_region.get_region_mean())
        nearest_neighbor_region_index = theta.argmin()
        print 'nearest_neighbor_region_index: ', nearest_neighbor_region_index, '   theta.min: ', theta.min()
        nearest_region = value_region_dicts[neighbor_region_values[nearest_neighbor_region_index]]

        return nearest_region

    def compute(self, image, mask):
        value_region_dicts, unique_image = self.convert_image_to_regions(image, mask)
        region_size = 0
        #Suppose the seed only contain one value
        seed_value = unique_image[self.seed[:, 0], self.seed[:, 1],self.seed[:, 2]]
        print 'value: ', seed_value[0]
        #may the seed_value contains some values, but here just pick the first one
        result_region = Region(seed_value[0], image, unique_image)
        while region_size < self.stop_size:
            #compute the nearest region
            nearest_region = self.compute_nearest_region(result_region, value_region_dicts)
            #update the result_region
            result_region.remove_neighbor_region(nearest_region)
            result_region.add_region(nearest_region)
            result_region.add_neighbor_region(nearest_region)
            #compute the stop condition
            region_size = result_region.get_region_size()
            print 'region_size: ', region_size

        return result_region

















