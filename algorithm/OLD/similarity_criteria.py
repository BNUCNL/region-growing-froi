# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""
The object to compute the similarity between the labeled region and its neighbors

"""

from scipy.spatial import distance
from algorithm.unsed.neighbor import *


class SimilarityCriteria(object):
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

    def compute_neighbor_distance(self, region, image, nbidx, metric=None):
        """
        Compute the distance between the labeled region and its neighbors.
        Parameters
        ----------
        image: numpy 2d/3d/4d array
            The numpy array to represent 2d/3d/4d image to be segmented.
        region: class Region
            represent the current region and associated attributes
        nbidx: numpy 1d array
            A array to provide the index which neighbors should be considered
        metric: str, optional
            distance metric
        """

        if metric is None:
            metric = self.metric

        if metric == 'euclidean':
            # compute distance for 2d image
            if image.ndim == 2:
                region_val = np.mean(image[region.label[:, 0], region.label[:, 1]])
                neighbor_val = image[region.neighbor[nbidx, 0], region.neighbor[nbidx, 1]]

                dist = np.abs(region_val - neighbor_val)

            # compute distance for 3d image
            elif image.ndim == 3:
                region_val = np.mean(image[region.label[:, 0], region.label[:, 1], region.label[:, 2]])
                neighbor_val = image[region.neighbor[nbidx, 0], region.neighbor[nbidx, 1], region.neighbor[nbidx, 2]]

                dist = np.abs(region_val - neighbor_val)

            # compute distance for 4d image
            else:
                region_val = np.mean(image[region.label[:, 0], region.label[:, 1], region.label[:, 2], :])
                neighbor_val = image[region.neighbor[nbidx, 0], region.neighbor[nbidx, 1], region.neighbor[nbidx, 2], :]

                dist = distance.cdist(region_val, neighbor_val, 'euclidean')

        elif metric == 'mahalanobis':
            if image.ndim == 2:
                region_val = np.mean(image[region.label[:, 0], region.label[:, 1]])
                region_std = np.std(image[region.label[:, 0], region.label[:, 1]])
                neighbor_val = image[region.neighbor[nbidx, 0], region.neighbor[nbidx, 1]]

                dist = np.abs(region_val - neighbor_val) / region_std

            # compute distance for 3d image
            elif image.ndim == 3:
                region_val = np.mean(image[region.label[:, 0], region.label[:, 1], region.label[:, 2]])
                region_std = np.std(image[region.label[:, 0], region.label[:, 1], region.label[:, 2]])
                neighbor_val = image[region.neighbor[nbidx, 0], region.neighbor[nbidx, 1], region.neighbor[nbidx, 2]]

                dist = np.abs(region_val - neighbor_val) / region_std

            # compute distance for 4d image
            elif image.ndim == 4:
                region_val = np.mean(image[region.label[:, 0], region.label[:, 1], region.label[:, 2], :])
                neighbor_val = image[region.neighbor[nbidx, 0], region.neighbor[nbidx, 1], region.neighbor[nbidx, 2], :]
                dist = distance.cdist(region_val, neighbor_val, 'mahalanobis')

        elif metric == 'exp':
            if image.ndim == 2:
                neighbor_val = image[region.neighbor[nbidx, 0], region.neighbor[nbidx, 1]]
                print neighbor_val

            # compute distance for 3d image
            elif image.ndim == 3:
                neighbor_val = image[region.neighbor[nbidx, 0], region.neighbor[nbidx, 1], region.neighbor[nbidx, 2]]

            dist = np.exp(-neighbor_val)

        return dist

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

    def __init__(self, prior_image, weight_method_name='PB', prior_weight=None, metric='euclidean', rand_neighbor_prop=1):
        """
        Parameters
        prior_image: np array
            the prior image used in the similarity computing
        weight_method_name: str, optional
            the weighted method for the prior,supporting probability based(PB) and distance based(DB) method
        prior_weight: double,
            the weight to use the prior
        metric: str,optional
            the distance metric such as 'euclidean', 'mahalanobis','exp'. Default is 'euclidean'.
        rand_neighbor_prop: double, optional
            the proportion of neighbors in calculating the similarity
        """
        super(PriorBasedSimilarityCriteria, self).__init__(metric, rand_neighbor_prop)

        if weight_method_name == 'PB' or weight_method_name == 'DB':
            self.weight_method_name = weight_method_name
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


    def set_weight(self, weight_method_name):
        """
        Set weighted method: PB(probability based) or DB(distance based)
        """
        self.weight_method_name = weight_method_name

    def get_weight(self):
        """
        Get weighted method.
        """
        return self.weight_method_name

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
        if 'PB' == self.weight_method_name:
            prior_dist = self.compute_neighbor_distance(region, prior_image, nbidx, 'exp')
            dist = image_dist * prior_dist
        else:
            prior_dist = self.compute_neighbor_distance(region, prior_image, nbidx, 'mahalanobis')
            dist = image_dist + prior_weight * prior_dist

        nearest_neighbor = region.neighbor[nbidx[dist.argmin()], :]
        return nearest_neighbor.reshape((-1, 3))

class SlicBasedSimilarityCriteria(SimilarityCriteria):
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


class MultiSeedsSimilarityCriteria(SimilarityCriteria):
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
        self.metric = metric

    def compute(self, regions, image, ssl):
        min_delta_key = ssl.keys()[np.array(ssl.values()).argmin()]
        nearest_neighbor_cord = np.array(min_delta_key)

        return min_delta_key, nearest_neighbor_cord









