import copy

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
        region = copy.copy(region)
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
            regions.append(copy.copy(region))
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
            regions.append(copy.copy(reg))
            self.stop_criteria.set_stop()

        return regions
