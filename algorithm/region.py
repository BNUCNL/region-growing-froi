# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""
An object to represent the region and its associated attributes

"""
import numpy as np

class Region(object):
    def __init__(self, seed, image, unique_image):
        if isinstance(seed, np.ndarray):
            seed = seed.tolist()
        self.seed = seed

        self.image = image
        self.unique_image = unique_image
        self.region_values = set()
        self.neighbor_values = set()

    def add_region(self, region):
        self.region_values |= region.get_region_value()

    def remove_region(self, region_value):
        self.region_values.remove(region_value)

    def get_region_value(self):
        self.region_values

    def add_neighbor_region(self, neighbor_value):
        self.neighbor_values.add(neighbor_value)

    def remove_neighbor_region(self, neighbor_value):
        self.neighbor_values.remove(neighbor_value)

    def get_region_mean(self):
        mask = self.generate_region_mask()
        self.mean = self.image[mask].mean()

    def get_region_values_size(self):
        return self.region_values.__len__()

    def get_region_size(self):
        mask = self.generate_region_mask()

        return mask.sum()

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




