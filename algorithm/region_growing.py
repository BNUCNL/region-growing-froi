# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""
Seeded region growing performs a segmentation of an image with respect to a set of points, known as seeded region.

"""

import copy
import numpy as np

class SeededRegionGrowing(object):
    """
    Seeded region growing performs a segmentation of an image with respect to a set of points, known as seeded region.
    Attributes
    similarity_criteria: SimilarityCriteria object
        The similarity criteria which control the neighbor to merge to the region
    stop_criteria: StopCriteria object
        The stop criteria which control when the region growing stop
    Methods
    compute(region,image)
        do region growing
    """

    def __init__(self, similarity_criteria, stop_criteria):
        """
        Initialize the object
        Parameters
        similarity_criteria: SimilarityCriteria object
            The similarity criteria which control the neighbor to merge to the region
        stop_criteria: StopCriteria object
            The stop criteria which control when the region growing stop
        """
        self.similarity_criteria = similarity_criteria
        self.stop_criteria = stop_criteria

    def set_stop_criteria(self, stop_criteria):
        """
        Set the stop criteria.
        """
        self.stop_criteria = stop_criteria

    def get_stop_criteria(self):
        """
        Return the stop criteria.
        """
        return self.stop_criteria

    def set_similarity_criteria(self, similarity_criteria):
        """
        Set the similarity criteria.
        """
        self.similarity_criteria = similarity_criteria

    def get_similarity_criteria(self):
        """
        Get the similarity criteria.
        """
        return self.similarity_criteria

    def compute(self, region, image, threshold):
        """
        Region grows based on the attributes seeds,similarity and stop criterion
        Parameters
        region: Region object
            seeded region for growing
        image: numpy 2d/3d/4d array
            The numpy array to represent 2d/3d/4d image to be segmented. In 4d image, the first 3D is spatial dimension
            and the 4th dimension is time or feature dimension
        threshold: numpy 1d array, float
            Stop thresholds for region growing
        Returns
            region: a list of region object
                The regions are generated for each threshold, one region for one threshold
        """
        regions = []
        region = copy.deepcopy(region)
        for thr in threshold:
            while not self.stop_criteria.isstop():
                # find the nearest neighbor for the current region
                nearest_neighbor = self.similarity_criteria.compute(region, image)
                # add the nearest neighbor to the region
                region.add_label(nearest_neighbor)
                # remove the nearest neighbor from the current neighbor
                region.remove_neighbor(nearest_neighbor)
                # compute the neighbor of the new added pixel and put it into the current neighbor
                region.add_neighbor(nearest_neighbor)
                # Update the stop criteria
                self.stop_criteria.compute(region, image, thr)
            regions.append(copy.deepcopy(region))
            self.stop_criteria.set_stop()

        return regions

class RandomSRG(SeededRegionGrowing):
    """
    Seeded region growing based on random seeds.
    Attributes
    similarity_criteria: SimilarityCriteria object
        The similarity criteria which control the neighbor to merge to the region
    stop_criteria: StopCriteria object
        The stop criteria which control when the region growing stop
    seed_sampling_num: int, optional
        The sampling number for seed with replacement
    Methods
    computing(image,region,threshold)
        Do region growing
    """

    def __init__(self, similarity_criteria, stop_criteria, seed_sampling_num, mask_image=None):
        """
        Initialize the object
        Parameters
        similarity_criteria: class SimilarityCriteria
            The similarity criteria which control the neighbor to merge to the region
        stop_criteria: class StopCriteria
            The stop criteria which control when the region growing stop
        """
        self.similarity_criteria = similarity_criteria
        self.stop_criteria = stop_criteria
        self.seed_sampling_num = seed_sampling_num
        self.mask_image = mask_image

    def compute(self, region, image, threshold):
        """
        Aggregation for different regions
        Parameters
        region: Region object
            seeded region for growing
        image: numpy 2d/3d/4d array
            The numpy array to represent 2d/3d/4d image to be segmented. In 4d image, the first 3D is spatial dimension
            and the 4th dimension is time or feature dimension
        threshold: a numpy nd array
            Stop threshold for growing
        Returns
        regions:  a 2D list of Region object
            The regions are generated for each seed and each threshold. As a result, regions are a 2D list.
            The first dim is for seeds, and the second dim is for threshold
        """
        regions = []
        coords = region.seed_sampling(self.seed_sampling_num)
        for seed in coords:
            region.set_seed(seed.reshape((-1, 3)))
            reg = super(RandomSRG, self).compute(region, image, threshold)
            regions.append(copy.deepcopy(reg))
            self.stop_criteria.set_stop()

        return regions

class SlicSRG(SeededRegionGrowing):
    """
    Seeded region growing based on random seeds.
    Attributes
    similarity_criteria: SimilarityCriteria object
        The similarity criteria which control the neighbor to merge to the region
    stop_criteria: StopCriteria object
        The stop criteria which control when the region growing stop
    seed_sampling_num: int, optional
        The sampling number for seed with replacement
    Methods
    computing(image,region,threshold)
        Do region growing
    """

    def __init__(self, similarity_criteria, stop_criteria, n_segmentation = 10000, mask_image=None):
        """
        Initialize the object
        Parameters
        similarity_criteria: class SimilarityCriteria
            The similarity criteria which control the neighbor to merge to the region
        stop_criteria: class StopCriteria
            The stop criteria which control when the region growing stop
        """
        self.similarity_criteria = similarity_criteria
        self.stop_criteria = stop_criteria
        self.n_segmentation = n_segmentation
        self.mask_image = mask_image

    def set_n_segmentation(self, n_segmentation):
        self.n_segmentation = n_segmentation

    def get_n_segmentation(self):
        return self.n_segmentation

    def get_slic_image(self, image):
        from skimage.segmentation import slic

        gray_image = (image - image.min()) * 255 / (image.max() - image.min())
        slic_image = slic(gray_image.astype(np.float),
                          n_segments=self.n_segmentation,
                          slic_zero=True,
                          sigma=2,
                          multichannel=False,
                          enforce_connectivity=True)
        return slic_image

    # def compute(self, region, image, threshold):
    #     return super(SlicSRG, self).compute(region, image, threshold)


class MultiSeedsSRG(SeededRegionGrowing):
    """
    Seeded region growing performs a segmentation of an image with respect to a set of points, known as seeded region.
    Attributes
    similarity_criteria: SimilarityCriteria object
        The similarity criteria which control the neighbor to merge to the region
    stop_criteria: StopCriteria object
        The stop criteria which control when the region growing stop
    Methods
    compute(region,image)
        do region growing
    """

    def __init__(self, similarity_criteria, stop_criteria):
        """
        Initialize the object
        Parameters
        similarity_criteria: SimilarityCriteria object
            The similarity criteria which control the neighbor to merge to the region
        stop_criteria: StopCriteria object
            The stop criteria which control when the region growing stop
        """
        self.similarity_criteria = similarity_criteria
        self.stop_criteria = stop_criteria

        #dict: key - neighbor_cord, value - neighbor_delta
        self.ssl = {}
        #boundary -1 ,  unlabel 0, label 1...n
        self.boundary = []

    def get_ssl(self):
        return self.ssl

    def clear_ssl(self):
        self.ssl.clear()

    def get_boundary(self):
        return self.boundary

    def add_ssl_element(self, cord, label):
        self.ssl[cord] = label

    def add_boundary_element(self, cord):
        self.boundary.append(cord)

    def remove_ssl_element(self, key):
        del self.ssl[key]

    def init_ssl(self, regions, image):
        for i in range(len(regions)):
            neighbors = regions[i].get_neighbor()
            labels = regions[i].get_label()
            mean = image[labels[:, 0], labels[:, 1], labels[:, 2]].mean()
            for j in range(neighbors.shape[0]):
                if not self.ssl.has_key(tuple(neighbors[j, :])):
                    neighbor_value = image[tuple(neighbors[j, :])]
                    self.add_ssl_element(tuple(neighbors[j, :]), abs(neighbor_value - mean))
            regions[i].set_neighbor(None)

    def compute(self, multi_seeds_regions, image, threshold, neighbor_element):
        """
        Region grows based on the attributes seeds,similarity and stop criterion
        Parameters
        region: Region object
            seeded region for growing
        image: numpy 2d/3d/4d array
            The numpy array to represent 2d/3d/4d image to be segmented. In 4d image, the first 3D is spatial dimension
            and the 4th dimension is time or feature dimension
        threshold: numpy 1d array, float
            Stop thresholds for region growing
        Returns
            region: a list of region object
                The regions are generated for each threshold, one region for one threshold
        """
        result_regions = []
        regions = copy.deepcopy(multi_seeds_regions)
        result_image = np.zeros_like(image).astype(np.int)
        for i in range(len(regions)):
            labels = regions[i].get_label()
            result_image[labels[:, 0], labels[:, 1], labels[:, 2]] = i + 1 #mark the label
        self.init_ssl(regions, image)

        for thr in threshold:
            while not self.stop_criteria.isstop():
                # find the nearest neighbor for the current region
                min_delta_key, nearest_neighbor_cord = self.similarity_criteria.compute(regions, image, self.ssl)

                neighbors = neighbor_element.compute(nearest_neighbor_cord)
                neighbor_values = result_image[neighbors[:, 0], neighbors[:, 1], neighbors[:, 2]]

                unique_values = np.unique(neighbor_values)
                if len(unique_values) != 2:
                    self.add_boundary_element(nearest_neighbor_cord)
                    result_image[tuple(nearest_neighbor_cord)] = -1 #boundary label value
                else:
                    new_label = np.unique(neighbor_values)[1] #[0, new label]
                    result_image[tuple(nearest_neighbor_cord)] = new_label
                    #update the corresponding region mean
                    new_region_mean = image[result_image == new_label].mean()

                    for i in range(neighbors.shape[0]):
                        cord = neighbors[i, :]
                        value = result_image[tuple(cord)]

                        if value == 0 and not self.ssl.has_key(tuple(cord)) :
                            self.add_ssl_element(tuple(cord), abs(image[tuple(cord)] - new_region_mean))
                # add the nearest neighbor to the region
                regions[new_label - 1].add_label(nearest_neighbor_cord.reshape(-1, 3))
                self.remove_ssl_element(min_delta_key)

                # Update the stop criteria
                self.stop_criteria.compute(regions, self.get_ssl(), thr)

            self.stop_criteria.set_stop()
            result_regions.append(copy.deepcopy(regions))
            print 'thr: ', thr, ' finished.'
            # self.clear_ssl()

        return result_regions















