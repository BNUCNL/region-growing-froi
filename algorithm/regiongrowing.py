import random

import numpy as np
from scipy.spatial import distance

import utils



class Seeds(object):
    """
    An object hold the coordinates of the seeded points and randomly sample the points

    Attributes
    ----------
    coords: list of tuple coordinates [((x1,y1,z1),(x2,y2,z2)),()]
        The element of the list is a series tuples, which in turn is a series of tuple,
        each holding the coordinates of a point
    sampling_number: int
        The sampling number for random sampling

    """

    def __init__(self, coords, sampling_number=0):
        """

        Parameters
        ----------
        coords: list of tuple coordinates [((x1,y1,z1),(x2,y2,z2)),()]
            The element of the list is a series tuples, which in turn is a series of tuple,
            each holding the coordinates of a point
        sampling_number: int
            the sampling number for random sampling.

        """

        self.coords = coords
        self.sampling_number = sampling_number

    def set_sampling_number(self, sampling_number):

        self.sampling_number = sampling_number

    def set_coords(self, coords):

        self.coords = coords

    def get_coords(self):

        return self.coords

    def random_sampling(self):
        """
        Randomly sample coordinates from the initial coordinates.
        In each sampling, only one coordinate from each group of seeds will be sampled

        """
        sampling_coords = []
        for r in range(self.sampling_number):
            single_sampling = []
            for g in range(len(self.coords)):
                single_sampling.append(random.choice(self.coords[g]))
            sampling_coords.append(single_sampling)

        self.coords = sampling_coords


class SimilarityCriteria:
    """
    The object to compute the similarity between the labeled region and its neighbors


    Attributes
    ----------
    metric: str,optional
    A description for the metric type


    Methods
    ------
    compute()
        Do computing the similarity between the labeled region and its neighbors


    """

    def __init__(self, metric='educlidean'):
        """
        Parameters
        -----------------------------------------------------
        metric: 'euclidean', 'mahalanobis', 'minkowski','seuclidean', 'cityblock',ect. Default is 'euclidean'.
        """
        if not isinstance(metric, str):
            raise ValueError("The value of metric must be str type. ")

        self.metric = metric

    def set_metric(self, metric):
        """
        Get the metric of the  similarity...
        """
        self.metric = metric

    def get_metric(self):
        """
        Get the metric of the  similarity...
        """
        return self.metric

    def compute(self, region, image, prior_image=None):
        """
        Compute the similarity between the labeled region and neighbors.

        Parameters
        ----------
        image: numpy 2d/3d/4d array
            The numpy array to represent 2d/3d/4d image to be segmented.
        region: class Region
            represent the current region and associated attributes
        prior_image: numpy 2d/3d/4d array, optional
            A image to provide the prior or weight that a point belongs to the region

        """

        lsize = region.label_size
        nsize = region.neighbor_size

        if image.ndim == 2:
            region_val = np.mean(image[region.label[:lsize, 0], region.label[:lsize, 1]])
            neighbor_val = image[region.neighbor[:nsize, 0], region.neighbor[:nsize, 1]]
            dist = np.abs(region_val - neighbor_val)

        elif image.ndim == 3:
            region_val = np.mean(image[region.label[:lsize, 0], region.label[:lsize, 1], region.label[:lsize, 2]])
            neighbor_val = image[region.neighbor[:nsize, 0], region.neighbor[:nsize, 1], region.neighbor[:nsize, 2]]
            dist = np.abs(region_val - neighbor_val)

        else:
            region_val = np.mean(image[region.label[:lsize, 0], region.label[:lsize, 1], region.label[:lsize, 2], :])
            neighbor_val = image[region.neighbor[:nsize, 0], region.neighbor[:nsize, 1], region.neighbor[:nsize, 2], :]
            dist = distance.cdist(region_val, neighbor_val, self.metric)

        index = dist.argmin()
        return region.neighbor[index, :]


class StopCriteria(object):
    """
    The object to compute and determine whether the growing should stop

    Attributes
    ----------
    metric: str
        A description for the metric type
    stop: boolean
        Indicate the growing status: False or True
    threshold: float
        A value to represent a indicates providing the threshold at which growing should stop

    Methods
    -------
    compute(self, region, image)
        determine whether the growing should stop

    """

    def __init__(self, threshold, criteria_metric='size'):
        """
        Parameters
        ----------
        threshold: float
            The default is None which means the adaptive method will be used.
        criteria_metric: str, optional
            A description for the metric type. The supported types include 'homogeneity','size','gradient'.
            Default is 'size'

        """

        self.metric = criteria_metric
        self.threshold = threshold
        self.stop = False

    def set_metric(self, metric):
        """
        Get the name of the stop criteria..
        """
        self.metric = metric

    def get_metric(self):
        """
        Get the name of the stop criteria..
        """
        return self.metric


    def compute(self, region, image=None):
        """
        compute the metric of region according to the region and judge whether the metric meets the stop threshold

        Parameters
        ----------
        image: numpy 2d/3d/4d array
            The numpy array to represent 2d/3d/4d image to be segmented.
        region: class Region
            represent the current region and associated attributes

        """

        if self.metric == 'size':
            if region.label_size > self.threshold:
                self.stop = True

    def isstop(self):
        return self.stop


class Region(object):
    """
    An object to represent the region and its associated attributes

    Attributes
    ----------
    label: numpy 2d array
        The coordinates of the points which have been merged into the regions
    neighbor: numpy 2 array
        The coordinates of the points which is the neighbor of the merged region
    label_size: int
        size of the label of region
    neighbor_size: int
        size of the neighbor of the label

    Methods
    -------
    add_label(label)
        add the coordinates of points to the label of region
    add_neighbor(neighbor)
        add the coordinates of points to the neighbor of region
    remove_neighbor(label)
        remove the label points from the neighbor of the region
    """

    def __init__(self, label, neighbor):
        """

        Parameters
        ----------
        label: numpy 2d array
            Each row represents the coordinates for a pixels. Number of the rows is the number of pixels
        neighbor:numpy 2d array
            Each row represents the coordinates for a pixels

        """

        if not isinstance(label, np.ndarray):
            raise ValueError("The current region of the Region class must be ndarray type. ")

        if not isinstance(neighbor, np.ndarray):
            raise ValueError("The neighbor of the Region class must be ndarray type. ")

        buffer_size = 10000
        self.label = np.zeros((buffer_size, 3), dtype=int)
        self.neighbor = np.zeros((buffer_size, 3), dtype=int)

        self.label_size = label.shape[0]
        self.label[:self.label_size, :] = label

        self.neighbor_size = neighbor.shape[0]
        self.neighbor[:self.neighbor_size, :] = neighbor


    def set_label(self, label):
        """
        set the coordinates of the labeled pixes
        """

        self.label = label

    def get_label(self):
        """
        Get the the coordinates of the labeled pixels
        """

        return self.label

    def set_neighbor(self, neighbor):
        """
        Set the coordinates of region neighbor.
        """
        self.neighbor = neighbor

    def get_neighbor(self):
        """
        Get the coordinates of region neighbor.
        """
        return self.neighbor

    def add_label(self, label):
        """
        Add the coordinates of new label to the label of region.

        Parameters
        ----------
        neighbor: numpy 2d array
            Each row represents the coordinates for a pixels
        """

        self.label[self.label_size, :] = label
        self.label_size += 1

    def add_neighbor(self, neighbor):
        """
        Add the coordinates of new neighbor to the neighbor of region.

        Parameters
        ----------
        neighbor: numpy 2d array
            Each row represents the coordinates for  a pixels
        """

        self.neighbor = neighbor
        # print utils.in2d(new_neighbor, self.neighbor[:self.neighbor_size, :])
        marked = np.logical_or(utils.in2d(neighbor, self.neighbor[:self.neighbor_size, :]),
                               utils.in2d(neighbor, self.label[:self.label_size, :]))

        neighbor = np.delete(neighbor, np.nonzero(marked), axis=0)

        # print new_neighbor

        self.neighbor[self.neighbor_size:(self.neighbor_size + neighbor.shape[0]), :] = neighbor
        self.neighbor_size = self.neighbor_size + neighbor.shape[0]

    def remove_neighbor(self, label):
        """
        Remove the coordinates of new label from the neighbor of region.

        Parameters
        ----------
        neighbor: numpy 2d array
            Each row represents the coordinates for a pixels
         """

        idx = np.nonzero(utils.in2d(self.neighbor[:self.neighbor_size, :], label))[0]
        self.neighbor = np.delete(self.neighbor, idx, 0)
        self.neighbor_size -= len(idx)


def compute_inner_boundary(self):
    """
        Compute the inner boundary
    """
    #Do something here.

    pass


def compute_external_boundary(self):
    """
        Compute the external boundary
    """
    #Do something here.
    pass


class SeededRegionGrowing:
    """
    Seeded region growing performs a segmentation of an image with respect to a set of points, known as seeds.


    Attributes
    ----------
    image: numpy 2d/3d/4d array
        The numpy array to represent 2d/3d/4d image to be segmented. In 4d image, the first three dimension is spatial dimension and
        the fourth dimension is time or feature dimension
    seeds: class Seeds
        The seeds at which region growing begin
    similarity_criteria: class SimilarityCriteria
        The similarity criteria which control the neighbor to merge to the region
    stop_criteria: class StopCriteria
        The stop criteria which control when the region growing stop
    neighbor:class SpatialNeighbor
        The neighbor generator which generate the spatial neighbor(coordinates)for a point

    Methods
    -------
    grow()
        do region growing

    """

    def __init__(self, image, seeds, similarity_criteria, stop_criteria, neighbor):
        """
        Initialize the object

        Parameters
        ----------
        image: numpy.array
            a 2d/3d/4d image to be segmentated
        seeds: class Seeds
            The seeds at which region growing begin
        similarity_criteria: class SimilarityCriteria
            The similarity criteria which control the neighbor to merge to the region
        stop_criteria: class StopCriteria
            The stop criteria which control when the region growing stop
        neighbor:class SpatialNeighbor
            The neighbor generator which generate the spatial neighbor(coordinates)for a point

        """

        if 2 <= len(image.shape) <= 4:
            self.image = image
        else:
            raise ValueError("Target image must be a 2D/3D/4D array.")

        self.seeds = seeds
        self.similarity_criteria = similarity_criteria
        self.stop_criteria = stop_criteria
        self.neighbor = neighbor

        region_label = np.array(self.seeds.coords)
        region_neighbor = self.neighbor.compute(self.seeds.coords)
        self.region = Region(region_label, region_neighbor)

    def set_image(self, image):
        self.image = image

    def get_image(self):
        return self.image

    def set_seeds(self, seeds):
        self.seeds = seeds

    def get_seeds(self):
        return self.seeds

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

    def set_neighbor(self, neighbor):
        """
        Set the connectivity.
        """
        self.neighbor = neighbor

    def get_neighbor(self):
        """
        Get the connectivity.
        """
        return self.neighbor

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

    def set_region(self, region):
        """
        Set the region sequence.
        """
        self.region = region

    def get_region(self):
        """
        Get the region sequence..
        """
        return self.region

    def grow(self):
        """
        Do region growing based on the attributes seeds,similarity and stop criterion

        """

        while not self.stop_criteria.isstop():
            nearest_neighbor = self.similarity_criteria.compute(self.region, self.image)

            self.region.add_label(nearest_neighbor)

            self.region.remove_neighbor(nearest_neighbor)

            self.region.add_neighbor(self.neighbor.compute(nearest_neighbor))

            self.stop_criteria.compute(self.region, self.image)
        return self.region


class Optimizer(object):
    """
    Region optimizer.
    """

    def __init__(self, image, region):
        """
        Parameters
        -----------------------------------------------------
        region_sequence: region sequence
        opt_measurement: the optimize measurement.
        mask_image: the mask image may be used in the compute process. which should be a ndarray type.
        prior_image:the prior image may be used in the compute process. which should be a ndarray type.
        """


class Aggregator(object):
    """
    Seeded region growing based on random seeds.
    """

    def __init__(self, image, region, type='average'):
        """
        Parameters
        -----------------------------------------------------
        region_sequence: A series of regions.
        raw_image: raw image.
        aggregator_type: 'average', 'magnitude', 'homogeneity', default is 'average'.
        """

        pass


    def compute(self):
        """
        Aggregation for different regions
        """


class RandomSRG(SeededRegionGrowing):
    """
    Seeded region growing based on random seeds.
    """

    def __init__(self, image, seeds, similarity_criteria, stop_criteria, neighbor, aggregator):
        """
        Parameters
        -----------------------------------------------------
        n_seeds: n seeds.
        stop_criteria: stop criteria about the n regions from n seeds.
        """
        super(RandomSRG, self).__init__(image, seeds, similarity_criteria, stop_criteria, neighbor)
        self.aggregator = aggregator


    def grow(self):
        """
        Aggregation for different regions
        """


class AdaptiveSRG(SeededRegionGrowing):
    """
    Adaptive seeded region growing.
    """

    def __init__(self, image, seeds, similarity_criteria, stop_criteria, neighbor, optimizer):
        super(AdaptiveSRG, self).__init__(image, seeds, similarity_criteria, stop_criteria, neighbor)
        self.optimizer = optimizer


    def grow(self):
        """
        Adaptive region growing.
        """


if __name__ == "__main__":
    seed_coors = (((1, 2, 3), (3, 2, 1)), ((4, 5, 6), (6, 5, 1)))
    seeds3d = Seeds(seed_coors)
    print seeds3d.generating()


    #similarity_criteria = NeighborSimilarity(metric='euclidean',)
    #stop_criteria = StopCriteria(name='region_size', threshold=300)
    #connectivity = Connectivity('6')

