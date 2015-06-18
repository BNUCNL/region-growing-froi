from scipy.spatial import distance
from algorithm.neighbor import *


class SimilarityCriteria:
    """
    The object to compute the similarity between the labeled region and its neighbors
    Attributes
    metric: str,optional
    A description for the metric type
    Methods
    compute(region, image, prior_image=None)
        Do computing the similarity between the labeled region and its neighbors
    """

    def __init__(self, metric='educlidean', rand_neighbor_prop=1):
        """
        Parameters
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


class PriorBasedSimilarityCriteria(SimilarityCriteria):
    """
    The object to compute the similarity between the labeled region and its neighbors
    Attributes
    metric: str,optional
    A description for the metric type
    Methods
    compute(region, image)
        Do computing the similarity between the labeled region and its neighbors
    """

    def __init__(self, prior_image, wei_meth='PB', prior_weight=None, metric='euclidean', rand_neighbor_prop=1):
        """
        Parameters
        prior_image: np array
            the prior image used in the similarity computing
        wei_meth: str, optional
            the weighted method for the prior,supporting probability based(PB) and distance based(DB) method
        prior_weight: double,
            the weight to use the prior
        metric: str,optional
            the distance metric such as 'euclidean', 'mahalanobis','exp'. Default is 'euclidean'.
        rand_neighbor_prop: double, optional
            the proportion of neighbors in calculating the similarity
        """
        super(PriorBasedSimilarityCriteria, self).__init__(metric, rand_neighbor_prop)

        if wei_meth == 'PB' or wei_meth == 'DB':
            self.wei_meth = wei_meth
        else:
            raise ValueError("The weighted method should be 'PB(probability based)' or 'DB(distance based)'.")

        self.prior_image = prior_image
        self.prior_weight = prior_weight

    def set_prior_image(self, prior_image):
        """
        Set prior image.
        """
        self.prior_image = prior_image

    def get_prior_image(self):
        """
        Get prior image.
        """
        return self.prior_image

    def set_prior_weight(self, prior_weight):
        """
        Set prior weight.
        """
        self.prior_weight = prior_weight

    def get_prior_weight(self):
        """
        Get prior weight.
        """
        return self.prior_weight


    def set_wei_meth(self, wei_meth):
        """
        Set weighted method: PB(probability based) or DB(distance based)
        """
        self.wei_meth = wei_meth

    def get_wei_meth(self):
        """
        Get weighted method.
        """
        return self.wei_meth

    def compute(self, region, image):
        """
        Compute the similarity between the labeled region and its neighbors.
        Parameters
        image: numpy 2d/3d/4d array
            The numpy array to represent 2d/3d/4d image to be segmented.
        region: class Region
            represent the current region and associated attributes
        """

        nsize = region.neighbor_size()
        prior_weight = self.get_prior_weight()
        prior_image = self.get_prior_image()

        if image.shape != self.prior_image.shape:
            raise ValueError("The image shape should match the prior image shape. ")

        if self.rand_neighbor_prop == 1:
            nbidx = np.arange(nsize)
        else:
            nbidx = np.random.choice(nsize, np.rint(nsize * self.rand_neighbor_prop), replace=False)

        image_dist = self.compute_neighbor_distance(region, image, nbidx)

        #  The distance include data distance and prior distance
        if 'PB' == self.wei_meth:
            prior_dist = self.compute_neighbor_distance(region, prior_image, nbidx, 'exp')
            dist = image_dist * prior_dist
        else:
            prior_dist = self.compute_neighbor_distance(region, prior_image, nbidx, 'mahalanobis')
            dist = image_dist + prior_weight * prior_dist

        nearest_neighbor = region.neighbor[nbidx[dist.argmin()], :]
        return nearest_neighbor.reshape((-1, 3))

class SlicBasedSimilarityCriteria:
    """
    The object to compute the similarity between the labeled region and its neighbors
    Attributes
    metric: str,optional
    A description for the metric type
    Methods
    compute(region, image, prior_image=None)
        Do computing the similarity between the labeled region and its neighbors
    """

    def __init__(self, metric='educlidean'):
        """
        Parameters
        metric: 'euclidean', 'mahalanobis', 'minkowski','seuclidean', 'cityblock',ect. Default is 'euclidean'.
        rand_neighbor_prop: Tge proportion of neighbors in calculating the similarity
        """
        if not isinstance(metric, str):
            raise ValueError("The metric must be a str. ")

        self.metric = metric

    def compute(self, region, image):
        neighbor_values = region.get_neighbor_value()
        neighbor_region_means = np.zeros((len(neighbor_values), ))
        slic_image = region.get_slic_image()
        for i in range(len(neighbor_values)):
            neighbor_region_means[i] = image[slic_image == neighbor_values[i]].mean()

        nearest_neighbor = neighbor_values[neighbor_region_means.argmax()]

        return nearest_neighbor



