import copy

from scipy.spatial import distance

from algorithm.neighbor import *


class Seeds(object):
    """
    An object hold the coordinates of seeded points and randomly sample the points

    Attributes
    ----------
    coords: list of tuple coordinates [(x1,y1,z1),(x2,y2,z2),]
        The element of the list is a series tuples, each of which in turn holds the coordinates of a point
    """

    def __init__(self, coords):
        """

        Parameters
        ----------
        coords: list of tuple coordinates [(x1,y1,z1),(x2,y2,z2),()]
            The element of the list is a series tuples, which in turn holds the coordinates of a point

        """

        if not isinstance(coords, np.ndarray):
            self.coords = np.array(coords)
        else:
            self.coords = coords


    def set_coords(self, coords):

        self.coords = coords

    def get_coords(self):

        return self.coords


class Region(object):
    """
    An object to represent the region and its associated attributes

    Attributes
    ----------
    label: numpy 2d array
        The coordinates of the points which have been merged into the regions
    neighbor: numpy 2 array
        The coordinates of the points which is the neighbor of the merged region
    boundary: numpy 2d array
        The coordinates of the points which is the boundary of current label


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
        neighbor: numpy 2d array
            Each row represents the coordinates for a pixels

        """

        if not isinstance(label, np.ndarray):
            raise ValueError("The current region of the Region class must be ndarray type. ")

        if not isinstance(neighbor, np.ndarray):
            raise ValueError("The neighbor of the Region class must be ndarray type. ")

        self.label = label
        self.neighbor = neighbor
        self.boundary = None


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

    def get_label_size(self):

        return self.label.shape[0]

    def get_neighbor_size(self):

        return self.neighbor.shape[0]

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
        #print self.label.shape, label.shape
        self.label = np.append(self.label, label, axis=0)


    def add_neighbor(self, neighbor):
        """
        Add the coordinates of new neighbor to the neighbor of region.

        Parameters
        ----------
        neighbor: numpy 2d array
            Each row represents the coordinates for  a pixels
        """

        # find the neighbor which have been in neighbor or in label list
        marked = np.logical_or(utils.in2d(neighbor, self.neighbor),
                               utils.in2d(neighbor, self.label))

        # delete the marked neighbor
        neighbor = np.delete(neighbor, np.nonzero(marked), axis=0)

        # Add unmarked neighbor to the region neighbor and update the neighbor size
        self.neighbor = np.append(self.neighbor, neighbor, axis=0)


    def remove_neighbor(self, label):
        """
        Remove the coordinates of label from the neighbor of region.

        Parameters
        ----------
        neighbor: numpy 2d array
            Each row represents the coordinates for a pixels
         """

        # find the index of the new added labels in the region neighbor list
        idx = np.nonzero(utils.in2d(self.neighbor, label))[0]
        self.neighbor = np.delete(self.neighbor, idx, 0)

    def compute_boundary(self, spatial_neighbor):

        """
            Compute the  boundary for the label
        """

        boundary = np.zeros(self.label.shape[0])
        for v in range(self.label.shape[0]):
            nb = spatial_neighbor.compute(self.label[v, :])
            if not np.all(utils.in2d(nb, self.label)):
                boundary[v] = True

        self.boundary = self.label[boundary, :]

    def get_boundary(self):
        """
        Get the boundary for the label
        """

        return self.boundary


class SimilarityCriteria:
    """
    The object to compute the similarity between the labeled region and its neighbors

    Attributes
    ----------
    metric: str,optional
    A description for the metric type

    Methods
    ------
    compute(region, image, prior_image=None)
        Do computing the similarity between the labeled region and its neighbors


    """

    def __init__(self, metric='educlidean', rand_neighbor_prop=1):
        """
        Parameters
        -----------------------------------------------------
        metric: 'euclidean', 'mahalanobis', 'minkowski','seuclidean', 'cityblock',ect. Default is 'euclidean'.
        rand_neighbor_prop: Tge proportion of neighbors in calculating the similarity
        """
        if not isinstance(metric, str):
            raise ValueError("The metric must be a str. ")

        self.metric = metric

        if 0 < rand_neighbor_prop <= 1:
            self.rand_neighbor_prop = rand_neighbor_prop
        else:
            raise ValueError("The rand_neighbor_prop must be between 0 and 1. ")

    def set_metric(self, metric):
        """
        Set the metric of the  similarity
        """
        self.metric = metric

    def get_metric(self):
        """
        Get the metric of the  similarity
        """
        return self.metric


    def set_rand_neighbor_prop(self, rand_neighbor_prop):
        """
        Set the neighbor prop for randomization
        """
        self.rand_neighbor_prop = rand_neighbor_prop

    def get_rand_neighbor_prop(self):
        """
        Get the neighbor prop for randomization
        """
        return self.rand_neighbor_prop

    def compute(self, region, image):
        """
        Compute the similarity between the labeled region and its neighbors.

        Parameters
        ----------
        image: numpy 2d/3d/4d array
            The numpy array to represent 2d/3d/4d image to be segmented.
        region: class Region
            represent the current region and associated attributes
        prior_image: numpy 2d/3d/4d array, optional
            A image to provide the prior or weight that a point belongs to the region

        """

        lsize = region.get_label_size()
        nsize = region.get_neighbor_size()

        if self.rand_neighbor_prop == 1:
            nbidx = np.arange(nsize)
        else:
            nbidx = np.random.choice(nsize, np.rint(nsize * self.rand_neighbor_prop), replace=False)


        # compute distance for 2d image
        if image.ndim == 2:
            region_val = np.mean(image[region.label[:lsize, 0], region.label[:lsize, 1]])
            neighbor_val = image[region.neighbor[nbidx, 0], region.neighbor[nbidx, 1]]
            dist = np.abs(region_val - neighbor_val)

        # compute distance for 3d image
        elif image.ndim == 3:
            region_val = np.mean(image[region.label[:lsize, 0], region.label[:lsize, 1], region.label[:lsize, 2]])
            neighbor_val = image[region.neighbor[nbidx, 0], region.neighbor[nbidx, 1], region.neighbor[nbidx, 2]]
            dist = np.abs(region_val - neighbor_val)

        # compute distance for 4d image
        else:
            region_val = np.mean(image[region.label[:lsize, 0], region.label[:lsize, 1], region.label[:lsize, 2], :])
            neighbor_val = image[region.neighbor[nbidx, 0], region.neighbor[nbidx, 1], region.neighbor[nbidx, 2], :]
            dist = distance.cdist(region_val, neighbor_val, self.metric)

        return region.neighbor[nbidx[dist.argmin()], :]


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
        threshold: numpy 1d array  or a list or a tuple
            The default is None which means the adaptive method will be used.
        criteria_metric: str, optional
            A description for the metric type. The supported types include 'homogeneity','size','gradient'.
            Default is 'size'

        """

        self.metric = criteria_metric

        if not isinstance(threshold, np.ndarray):
            threshold = np.array(threshold)

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

    def set_stop(self, stop=False):
        """
        Set the stop status
        """
        self.stop = False

    def compute(self, region, image=None):
        """
        compute the metric of region according to the region and judge whether the metric meets the stop threshold

        Parameters
        ----------
        image: numpy 2d/3d/4d array
            The numpy array to represent 2d/3d/4d image to be segmented.
        region: Region object
            It represents the current region and associated attributes

        """

        if self.metric == 'size':
            if region.get_label_size() >= self.threshold:
                self.stop = True

    def isstop(self):
        return self.stop


class SeededRegionGrowing(object):
    """
    Seeded region growing performs a segmentation of an image with respect to a set of points, known as seeds.


    Attributes
    ----------
    image: numpy 2d/3d/4d array
        The numpy array to represent 2d/3d/4d image to be segmented. In 4d image, the first three dimension is spatial dimension and
        the fourth dimension is time or feature dimension
    seeds: Seeds object
        The seeds at which region growing begin
    similarity_criteria: SimilarityCriteria object
        The similarity criteria which control the neighbor to merge to the region
    stop_criteria: StopCriteria object
        The stop criteria which control when the region growing stop
    neighbor: SpatialNeighbor object
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
            a 2d/3d/4d image to be segmented
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

        # initialize the region
        region_label = self.seeds.coords
        region_neighbor = self.neighbor.compute(region_label)  # compute the neighbor for the current region(label)
        self.region = Region(region_label, region_neighbor)


    def set_image(self, image):
        self.image = image

    def get_image(self):
        return self.image

    def set_seeds(self, seeds):
        self.seeds = seeds
        self.region.set_label(self.seeds.get_coords())
        self.region.set_neighbor(self.neighbor.compute(self.seeds.get_coords()))

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
         Region grows based on the attributes seeds,similarity and stop criterion

         Returns
         -------
         region: Region object

        """

        while not self.stop_criteria.isstop():
            # find the nearest neighbor for the current region
            nearest_neighbor = self.similarity_criteria.compute(self.region, self.image)
            nearest_neighbor = nearest_neighbor.reshape((1, -1))

            # add the nearest neighbor to the region
            self.region.add_label(nearest_neighbor)

            # remove the nearest neighbor from the current neighbor
            self.region.remove_neighbor(nearest_neighbor)

            # compute the neighbor of the new added voxels and put it into the current neighbor
            self.region.add_neighbor(self.neighbor.compute(nearest_neighbor))

            # Update the stop criteria
            self.stop_criteria.compute(self.region, self.image)

            #print self.region.label.shape

        return self.region


class Aggregator(object):
    """
    Aggregator for a set of regions.

    Attributes
    ----------
    aggr_type:  str, optional
        Description for the method to  aggregate the regions. Supported methods include
        'uniform weighted average'(UWA), 'magnitude weighted average'(MWA), 'homogeneity  weighted _average'(HWA),
        default is DA.
    """

    def __init__(self, agg_type='UWA'):
        """
        Parameters
        ----------
        aggr_type:  str, optional
        Description for the method to  aggregate the regions. Supported methods include
        'uniform weighted average'(UWA), 'magnitude weighted average'(MWA), 'homogeneity  weighted _average'(HWA),
        default is DA.
        """
        self.agg_type = agg_type

    def compute(self, region, image):
        """
        Aggregate a set of regions

        Parameters
        ----------
        region: A list of regions.
            A set of regions to be aggregated
        image: numpy 2d/3d/4d/ array
            image to be  segmented.

        Returns
        -------
        agg_image: numpy 2d/3d array
            Final segmentation from aggregating a set of regions

        """
        if image.ndim == 2:
            shape = (image.shape[0], image.shape[1], len(region))
        elif image.ndim == 3 or image.ndim == 4:
            shape = (image.shape[0], image.shape[1], image.shape[2], len(region))
        else:
            raise ValueError("Wrong image dimension")

        print len(region)

        region_image = np.zeros((image.shape[0], image.shape[1], image.shape[2], len(region)), dtype=int)
        weight = np.ones(len(region))
        if self.agg_type == 'UWA':
            for r in range(len(region)):
                label = region[r].get_label()
                region_image[label[:, 0], label[:, 1], label[:, 2], r] = 1

        elif self.agg_type == 'MWA':
            for r in range(len(region)):
                label = region[r].get_label()
                region_image[label[:, 0], label[:, 1], label[:, 2], r] = 1

                weight[r] = np.mean(image[label[:, 0], label[:, 1], label[:, 2]])

        elif self.agg_type == 'HWA':
            for r in range(len(region)):
                label = region[r].get_label()
                region_image[label[:, 0], label[:, 1], label[:, 2], r] = 1
                weight[r] = np.std(image[label[:, 0], label[:, 1], label[:, 2]])

        else:
            raise ValueError("The Type of aggregator should be 'UWA', 'MWA', and 'HWA'.")

        weight = weight / weight.sum()
        agg_image = np.average(region_image, axis=3, weights=weight)

        return agg_image


class RandomSRG(object):
    """
    Seeded region growing based on random seeds.

        Attributes
    ----------
    image: numpy 2d/3d/4d array
        The numpy array to represent 2d/3d/4d image to be segmented. In 4d image, the first three dimension is spatial dimension and
        the fourth dimension is time or feature dimension
    seeds: Seeds object
        The seeds at which region growing begin
    similarity_criteria: SimilarityCriteria object
        The similarity criteria which control the neighbor to merge to the region
    stop_criteria: StopCriteria object
        The stop criteria which control when the region growing stop
    neighbor: SpatialNeighbor object
        The neighbor generator which generate the spatial neighbor(coordinates)for a point
    seed_sampling_num: int, optional
        The sampling number for seed with replacement

    Methods
    -------
    grow()
        do region growing

    """

    def __init__(self, srg, seed_sampling_num, aggregator=None):
        """

        Parameters
        ----------
        srg: SeedRegionGrowing object
        seed_sampling_num: int, optional
           The random sampling number for seeds
        aggregator: Aggregator object

        """

        self.srg = srg
        self.aggregator = aggregator

        if seed_sampling_num >= 0:
            self.seed_sampling_num = seed_sampling_num
        else:
            raise ValueError("The  seed_sampling_num must be a positive int")


    def seed_sampling(self):
        """
        Randomly sample coordinates from the initial coordinates.
        In each sampling, only one coordinate from each group of seeds will be sampled

        """

        seed_coords = self.srg.get_seeds().get_coords()

        if self.seed_sampling_num > 0:
            sampling_coords = seed_coords[np.random.choice(seed_coords.shape[0], self.seed_sampling_num, replace=True),
                              :]
        else:
            sampling_coords = seed_coords

        return sampling_coords


    def grow(self):
        """
        Aggregation for different regions


        Returns
        -------
        regions:  a list of Region object

        """
        rand_coords = self.seed_sampling()
        regions = []
        for seed in rand_coords:
            self.srg.set_seeds(Seeds(seed.reshape((1, -1))))
            regions.append(copy.copy(self.srg.grow()))
            self.srg.get_stop_criteria().set_stop(False)

        #region = self.aggregator.compute(regions, self.image)

        return regions


class Optimizer(object):
    """
    Optimizer to select the optimal segmentation from a set of region growing results.

    Attributes
    ----------
    opt_type:  str, optional
        Description for the criteria to select the optimal segmentation from region growing. methods include
        'peripheral contrast'(PC), 'average contrast'(AC), 'homogeneity  weighted _average'(HWA),
        default is PC.
    """


    def __init__(self, opt_type):
        """
        Parameters
        ----------
        opt_type:  str, optional
            Description for the criteria to select the optimal segmentation from region growing. methods include
            'peripheral contrast'(PC), 'average contrast'(AC), 'homogeneity  weighted _average'(HWA),
            default is PC.
        """

        self.opt_type = opt_type

    def compute(self, region, image):
        """
        Find the optimal segmentation according to the specified optimization criteria

        Parameters
        ----------
        region: A list of region object
            A set of regions to be aggregated
        image: numpy 2d/3d/4d/ array
            image to be  segmented.

        """

        if self.opt_type == 'PC':
            bound_val = np.mean(image[region.boundary[:, 0], region.boundary[:, 1], region.boundary[:, 2]])
            per_val = np.mean(image[region.neighbor[:, 0], region.neighborl[:, 1], region.neighbor[:, 2]])

            contrast = (bound_val - per_val) / (bound_val + per_val)
        elif self.opt_type == 'AC':
            region_val = np.mean(image[region.label[:, 0], region.label[:, 1], region.label[:, 2]])
            per_val = np.mean(image[region.neighbor[:, 0], region.neighborl[:, 1], region.neighbor[:, 2]])

            contrast = (region_val - per_val) / (region_val + per_val)
        else:
            raise ValueError("The Type of aggregator should be 'PC' and 'AC'.")


class AdaptiveSRG(object):
    """
    Adaptive seeded region growing.
    """

    def __init__(self, srg, threshold, optimizer=None):
        self.srg = srg
        self.threshold = threshold
        self.optimizer = optimizer

    def grow(self):
        """
        Adaptive region growing.
        """
        regions = []
        for t in self.threshold:
            self.srg.get_stop_criteria().set_threshod(t)
            regions.append(copy.copy(self.srg.grow()))
            self.srg.get_stop_criteria().set_stop(False)


if __name__ == "__main__":
    seedcoords = (((1, 2, 3), (3, 2, 1)), ((4, 5, 6), (6, 5, 1)))
    seeds3d = Seeds(seedcoords)
    print seeds3d.get_coords()

