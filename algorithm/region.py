# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""
An object to represent the region and its associated attributes

"""
import numpy as np

class Region(object):
    def __init__(self, value, image, unique_image):
        self.image = image
        self.unique_image = unique_image
        self.region_values = set()
        self.region_values.add(value)
        self.neighbor_values = set()

    def add_region(self, region):
        self.region_values |= region.get_region_value()

    def remove_region(self, region):
        self.region_values ^= region.get_region_value()

    def get_region_value(self):
        return self.region_values

    def add_neighbor_region(self, neighbor_region):
        self.neighbor_values |= neighbor_region.get_neighbor_region_value()
        self.neighbor_values -= self.region_values

    def remove_neighbor_region(self, neighbor_region):
        self.neighbor_values -= neighbor_region.get_region_value()

    def get_neighbor_region_value(self):
        return self.neighbor_values

    def get_region_mean(self):
        mask = self.generate_region_mask()
        self.mean = self.image[mask].mean()

        return self.mean

    def get_region_values_size(self):
        return self.region_values.__len__()

    def get_region_size(self):
        mask = self.generate_region_mask()
        return mask.sum()

    def get_neighbor_values_size(self):
        return self.neighbor_values.__len__()

    def get_neighbor_size(self):
        mask = self.generate_region_neighbor_mask()
        return mask.sum()

    def generate_region_mask(self):
        region_mask = np.zeros_like(self.unique_image).astype(np.bool)
        for region_value in self.region_values:
            region_mask[self.unique_image == region_value] = True

        return region_mask

    def generate_region_neighbor_mask(self):
        neighbor_mask = np.zeros_like(self.unique_image).astype(np.bool)
        for neighbor_value in self.neighbor_values:
            neighbor_mask[self.unique_image == neighbor_value] = True

        return neighbor_mask

    def get_region_cords(self):
        dimension = len(self.image.shape)
        region_mask = self.generate_region_mask()
        region_cords = np.nonezeros(region_mask).reshape((dimension, -1))

        return region_cords

    def get_neighbor_region_cords(self):
        dimension = len(self.image.shape)
        neighbor_region_mask = self.generate_region_neighbor_mask()
        neighbor_region_cords = np.nonezeros(neighbor_region_mask).reshape((dimension, -1))

        return neighbor_region_cords

    def compute_neighbor_regions(self):
        from scipy.ndimage.morphology import binary_dilation

        region_mask = self.generate_region_mask()
        #compute new neighbors
        neighbor_mask = binary_dilation(region_mask)
        neighbor_mask[region_mask] = 0
        neighbor_values = np.unique(self.unique_image[neighbor_mask > 0])
        neighbor_values = np.delete(neighbor_values, 0)
        #Add new neighbor values
        self.neighbor_values |= set(neighbor_values.tolist())







