import copy

from scipy.spatial import distance

from algorithm.neighbor import *


class Region(object):
    """
    An object to represent the region and its associated attributes

    Attributes
    ----------
    seed: list of tuple coordinates [(x1,y1,z1),(x2,y2,z2),]
        The element of the list is a series tuples, each of which in turn holds the coordinates of a point
    neighbor_element: SpatialNeighbor object
        The neighbor generator which generate the spatial neighbor(coordinates)for a point
    label: numpy 2d array
        The coordinates of the points which have been merged into the regions
    neighbor: numpy 2 array
        The coordinates of the points which is the neighbor of the merged region

    """

    def __init__(self, seed, neighbor_element):
        """

        Parameters
        ----------
        label: numpy 2d array
            Each row represents the coordinates for a pixels. Number of the rows is the number of pixels
        neighbor: numpy 2d array
            Each row represents the coordinates for a pixels

        """

        if not isinstance(seed, np.ndarray):
            seed = np.array(seed)

        self.seed = seed
        self.neighbor_element = neighbor_element

        self.label = seed
        self.neighbor = neighbor_element.compute(seed)

    def set_seed(self, seed):
        """
        set the coordinates of the seeds
        """
        if not isinstance(seed, np.ndarray):
            seed = np.array(seed)

        self.seed = seed
        self.label = seed
        self.neighbor = self.neighbor_element.compute(seed)
        #print seed.shape[0], self.neighbor.shape[0]

    def get_seed(self):
        """
        Get the the coordinates of the seeds
        """
        return self.seed

    def seed_sampling(self, sampling_num):
        """
        Randomly sample coordinates from the seeds.
        In each sampling, only one coordinate from each group of seeds will be sampled

        """
        if sampling_num > 0:
            sampling_seed = self.seed[np.random.choice(self.seed.shape[0], sampling_num, replace=True), :]
        else:
            sampling_seed = self.seed

        return sampling_seed

    def label_size(self):
        """
        Return the size of current label.
        """
        return self.label.shape[0]

    def neighbor_size(self):
        """
        Return the size of current neighbor.
        """

        return self.neighbor.shape[0]

    def set_neighbor_element(self, neighbor_element):
        """
        Set the coordinates of region neighbor.
        """
        self.neighbor_element = neighbor_element


    def get_neighbor_element(self, neighbor_element):
        """
        Set the coordinates of region neighbor.
        """
        return self.neighbor_element


    def add_label(self, label):
        """
        Add the coordinates of new label to the label of region.

        Parameters
        ----------
        neighbor: numpy 2d array
            Each row represents the coordinates for a pixels
        """
        self.label = np.append(self.label, label, axis=0)

        #self.label = np.array(self.label.tolist() + label.tolist())
        #self.label = np.array(np.append(self.label, label, axis=0).tolist())

        # print 'label', self.label.shape[0]


    def get_label(self):
        """
        Get the the coordinates of the labeled pixels
        """
        return self.label


    def get_neighbor(self):
        """
        Get the the coordinates of the neighbor pixels
        """
        return self.neighbor

    def add_neighbor(self, neighbor):
        """
        Add the coordinates of new neighbor to the neighbor of region.

        Parameters
        ----------
        neighbor: numpy 2d array
            Each row represents the coordinates for  a pixels
        """

        neighbor = self.neighbor_element.compute(neighbor)

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

    def compute_boundary(self):

        """
            Compute the  boundary for the label
        """

        boundary = np.zeros(self.label.shape[0]).astype(np.False_)
        for v in range(self.label.shape[0]):
            # nb = self.neighbor_element.compute(self.label[v, :])
            nb = self.neighbor_element.compute(self.label[v, :].reshape(1, len(self.label[v, :])))
            if not np.all(utils.in2d(nb, self.label)):
                boundary[v] = True

        return self.label[boundary, :]


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

        lsize = region.label_size()
        nsize = region.neighbor_size()

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

        nearest_neighbor = region.neighbor[nbidx[dist.argmin()], :]
        return nearest_neighbor.reshape((-1, 3))


class StopCriteria(object):
    """
    The object to compute and determine whether the growing should stop

    Attributes
    ----------
    metric: str
        A description for the metric type
    stop: boolean
        Indicate the growing status: False or True


    Methods
    -------
    compute(self, region, image,threshold)
        determine whether the growing should stop

    """

    def __init__(self, criteria_metric='size', stop=False):
        """
        Parameters
        ----------
        criteria_metric: str, optional
            A description for the metric type. The supported types include 'homogeneity','size','gradient'.
            Default is 'size'

        """
        if criteria_metric == 'size' or criteria_metric == 'homogeneity' or criteria_metric == 'gradient':
            self.metric = criteria_metric

        self.stop = stop

    def set_metric(self, metric):
        """
        Set the name of the stop criteria..
        """
        self.metric = metric

    def get_metric(self):
        """
        Get the name of the stop criteria..
        """
        return self.metric


    def compute(self, region, image, threshold=None):
        """
        compute the metric of region according to the region and judge whether the metric meets the stop threshold

        Parameters
        ----------
        image: numpy 2d/3d/4d array
            The numpy array to represent 2d/3d/4d image to be segmented.
        region: Region object
            It represents the current region and associated attributes
        threshold: float, optional
            The default is None which means the adaptive method will be used.

        """

        if self.metric == 'size':
            if region.label_size() >= threshold:
                self.stop = True

    def isstop(self):

        return self.stop

    def set_stop(self):
        """
        Reset the stop signal
        """
        self.stop = False


class SeededRegionGrowing(object):
    """
    Seeded region growing performs a segmentation of an image with respect to a set of points, known as seeded region.

    Attributes
    ----------
    similarity_criteria: SimilarityCriteria object
        The similarity criteria which control the neighbor to merge to the region
    stop_criteria: StopCriteria object
        The stop criteria which control when the region growing stop

    Methods
    -------
    compute(region,image)
        do region growing

    """

    def __init__(self, similarity_criteria, stop_criteria):
        """
        Initialize the object

        Parameters
        ----------
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
        ----------
        region: Region object
            seeded region for growing
        image: numpy 2d/3d/4d array
            The numpy array to represent 2d/3d/4d image to be segmented. In 4d image, the first 3D is spatial dimension
            and the 4th dimension is time or feature dimension
        threshold: numpy 1d array, float
            Stop thresholds for region growing


        Returns
        -------
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
    ----------
    similarity_criteria: SimilarityCriteria object
        The similarity criteria which control the neighbor to merge to the region
    stop_criteria: StopCriteria object
        The stop criteria which control when the region growing stop
    seed_sampling_num: int, optional
        The sampling number for seed with replacement

    Methods
    -------
    computing(image,region,threshold)
        Do region growing

    """

    def __init__(self, similarity_criteria, stop_criteria, seed_sampling_num):
        """
        Initialize the object

        Parameters
        ----------
        similarity_criteria: class SimilarityCriteria
            The similarity criteria which control the neighbor to merge to the region
        stop_criteria: class StopCriteria
            The stop criteria which control when the region growing stop

        """

        self.similarity_criteria = similarity_criteria
        self.stop_criteria = stop_criteria
        self.seed_sampling_num = seed_sampling_num


    def compute(self, region, image, threshold):
        """
        Aggregation for different regions

        Parameters
        ----------
        region: Region object
            seeded region for growing
        image: numpy 2d/3d/4d array
            The numpy array to represent 2d/3d/4d image to be segmented. In 4d image, the first 3D is spatial dimension
            and the 4th dimension is time or feature dimension
        threshold: a numpy nd array
            Stop threshold for growing

        Returns
        -------
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


class PriorSRG(SeededRegionGrowing):
    """
    Seeded region growing with prior information

    Attributes
    ----------
    similarity_criteria: SimilarityCriteria object
        The similarity criteria which control the neighbor to merge to the region
    stop_criteria: StopCriteria object
        The stop criteria which control when the region growing stop
    seed_sampling_num: int, optional
        The sampling number for seed with replacement
    prior_weight: weight for the prior
    prior_image: image with prior information for each voxels

    Methods
    -------
    computing(image,region,threshold)
        Do region growing

    """

    def __init__(self, similarity_criteria, stop_criteria, seed_sampling_num, prior_image, prior_weight):
        """
        Initialize the object

        Parameters
        ----------
        similarity_criteria: class SimilarityCriteria
            The similarity criteria which control the neighbor to merge to the region
        stop_criteria: class StopCriteria
            The stop criteria which control when the region growing stop

        """

        self.similarity_criteria = similarity_criteria
        self.stop_criteria = stop_criteria
        self.seed_sampling_num = seed_sampling_num
        self.prior_image = prior_image
        self.prior_weight = prior_weight

    def set_prior_image(self, prior_image):
        """
        Set prior image.
        """

        self.prior_image = prior_image

    def set_prior_weight(self, prior_weight):
        """
        Set prior weight.
        """
        self.prior_weight = prior_weight


    def compute(self, region, image, threshold):
        """
        Aggregation for different regions

        Parameters
        ----------
        region: Region object
            seeded region for growing
        image: numpy 2d/3d/4d array
            The numpy array to represent 2d/3d/4d image to be segmented. In 4d image, the first 3D is spatial dimension
            and the 4th dimension is time or feature dimension
        threshold: a numpy nd array
            Stop threshold for growing

        Returns
        -------
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
        default is UWA.
        """
        if agg_type == 'UWA' or agg_type == 'MWA' or agg_type == 'HWA':
            self.agg_type = agg_type
        else:
            raise ValueError("The Type of aggregator should be 'UWA', 'MWA', and 'HWA'.")

    def compute(self, region, image):
        """
        Aggregate a set of regions

        Parameters
        ----------
        region: A list of list of region objects to be aggregated..
            A 2d lists. the first dimension is for seeds, the second dimension is for threshold
        image: numpy 2d/3d/4d/ array
            image to be  segmented.

        Returns
        -------
        agg_image: numpy 2d/3d array
            Final segmentation from aggregating a set of regions

        """

        if image.ndim == 2:
            agg_image = np.zeros((image.shape[0], image.shape[1], len(region)), dtype=int)
            region_image = np.zeros((image.shape[0], image.shape[1], len(region[0])), dtype=int)
            weight = np.ones(len(region[0]))

            if self.agg_type == 'UWA':
                for i in range(len(region)):
                    for j in range(len(region[0])):
                        label = region[i][j].get_label()
                        region_image[label[:, 0], label[:, 1], j] = 1

                    weight = weight / weight.sum()
                    agg_image[:, :, i] = np.average(region_image, axis=2, weights=weight)
                    region_image[:] = 0

            elif self.agg_type == 'MWA':
                for i in range(len(region)):
                    for j in range(len(region[0])):
                        label = region[i][j].get_label()
                        region_image[label[:, 0], label[:, 1], j] = 1
                        weight[j] = np.mean(image[label[:, 0], label[:, 1]])

                    weight = weight / weight.sum()
                    agg_image[:, :, i] = np.average(region_image, axis=2, weights=weight)
                    region_image[:] = 0

            elif self.agg_type == 'HWA':
                for i in range(len(region)):
                    for j in range(len(region[0])):
                        label = region[i][j].get_label()
                        region_image[label[:, 0], label[:, 1], j] = 1
                        weight[j] = np.std(image[label[:, 0], label[:, 1]])

                    weight = weight / weight.sum()
                    agg_image[:, :, i] = np.average(region_image, axis=2, weights=weight)
                    region_image[:] = 0

        elif image.ndim == 3:
            agg_image = np.zeros((image.shape[0], image.shape[1], image.shape[2], len(region)), dtype=int)
            region_image = np.zeros((image.shape[0], image.shape[1], image.shape[2], len(region)), dtype=int)
            weight = np.ones(len(region))
            if self.agg_type == 'UWA':
                for i in range(len(region)):
                    for j in range(region[0]):
                        label = region[i][j].get_label()
                        region_image[label[:, 0], label[:, 1], label[:, 2], j] = 1

                    weight = weight / weight.sum()
                    agg_image[:, :, :, i] = np.average(region_image, axis=3, weights=weight)
                    region_image[:] = 0

            elif self.agg_type == 'MWA':
                for i in range(len(region)):
                    for j in range(len(region[0])):
                        label = region[i][j].get_label()
                        region_image[label[:, 0], label[:, 1], label[:, 2], j] = 1
                        weight[j] = np.mean(image[label[:, 0], label[:, 1], label[:, 2]])

                    weight = weight / weight.sum()
                    agg_image[:, :, :, i] = np.average(region_image, axis=3, weights=weight)
                    region_image[:] = 0

            elif self.agg_type == 'HWA':
                for i in range(len(region)):
                    for j in range(len(region[0])):
                        label = region[i][j].get_label()
                        region_image[label[:, 0], label[:, 1], label[:, 2], j] = 1
                        weight[j] = np.std(image[label[:, 0], label[:, 1], label[:, 2]])

                    weight = weight / weight.sum()
                    agg_image[:, :, :, i] = np.average(region_image, axis=3, weights=weight)
                    region_image[:] = 0

        elif image.ndim == 4:
            pass

        return agg_image


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

        if opt_type == 'PC' or opt_type == 'AC':
            self.opt_type = opt_type
        else:
            raise ValueError("The Type of aggregator should be 'PC' and 'AC'.")

    def compute(self, region, image):
        """
        Find the optimal segmentation according to the specified optimization criteria

        Parameters
        ----------
        region: A 2d list of region object
            A 2d lists. the first dimension is for seeds, the second dimension is for threshold
        image: numpy 2d/3d/4d/ array
            image to be  segmented.

        Returns
        ------
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
                        if (region_val + per_val) == 0:
                            con_val[i, j] = 0
                        else:
                            con_val[i, j] = (region_val - per_val) / (region_val + per_val)

        elif image.ndim == 4:
            pass

        return con_val


if __name__ == "__main__":
    pass
