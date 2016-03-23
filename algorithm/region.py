# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""
An object to represent the region and its associated attributes

"""
import numpy as np


class Region(object):
    def __init__(self, position, value, r_id, neighbor):
        """

        Parameters
        ----------
        position: coordinates array, n_voxel x 3 np.array
        value : value array, n_voxel x n_feature np.array
        id_: region id, a unique scalar
        neighbor: neighbor id array(n_neighbor, )

        Returns
        -------

        """
        self.position = position
        self.value = value
        self.id = r_id
        self.component = r_id
        self.neighbor = neighbor
        self.sum = np.mean(self.value)

    def add(self, region):
        if not np.any(self.component == region.id):
            self.position = np.append(self.position, region.position)
            self.value = np.append(self.value, region.value)
            self.sum = np.mean(self.value)
            self.component = np.append(self.component, region.id)

    def value(self):
        return self.sum

    def add_neighbor(self, region):
        if not np.any(self.neighbor == region.id):
            self.neighbor = np.append(self.neighbor, region.id)

    def remove_neighbor(self, region):
        self.neighbor = self.neighbor[self.neighbor != region.id]

    def size(self):
        return self.value.shape[0]

    def neighbor_size(self):
        return self.neighbor.shape[0]

    def nearest_neighbor(self):
        pass


class RepresentImageToRegion(object):

    """

    Parameters
    ----------
    image: image array
    meth: original or divided
    mask: mask to indicate the spatial extent to be represented

    Returns
    -------
    regions: a list of Region object

    """
    def __init__(self, meth='slic'):
        self.meth = meth

    def compute(self, image, n_region, mask=None):

        from skimage import segmentation, filter
        from skimage.future import graph

        gray_image = image.copy()
        if mask is not None:
            gray_image[mask > 0] = 0

        # Convert the original image to the 0~255 gray image
        gray_image = (gray_image - gray_image.min()) * 255 / (gray_image.max() - gray_image.min())
        labels = segmentation.slic(gray_image.astype(np.float),
                                   n_segments=n_region,
                                   slic_zero=True, sigma=2,
                                   multichannel=False,
                                   enforce_connectivity=True)
        edge_map = filter.sobel(gray_image)
        rag = graph.rag_boundary(labels, edge_map)
        rag = rag > 0

        regions = []
        for r in np.arange(rag.shape[0]):
            position = np.nonzero(labels == r)
            regions.append(Region(position, value=None, id_=None, neighbor=None))

        return regions

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
    def __int__(self, similarity_measure, stop_criteria):
        self.similarity_measure = similarity_measure
        self.stop_criteria = stop_criteria

    def compute(self, seed_region, candidate_region):
        """

        Parameters
        ----------
        seed_region: region list for seeds
        candidate_region: region list for candidate region

        Returns
        -------

        """

        n_seed = len(seed_region)
        region_size = np.zeros(n_seed)
        for r in np.arange(n_seed):
            region_size[r] = seed_region[r].size()

        dist = np.zeros(n_seed)
        neighbor =  np.zeros(n_seed)

        while np.any(np.less(region_size < self.stop_criteria)):
            r_index = np.less(region_size < self.stop_criteria)

            dist[r_index] = np.inf
            for r in np.count_nonzero(r_index):
                # find the nearest neighbor for the each seed region
                r_dist, r_neighbor = seed_region[r].nearest_neighbir()
                dist[r] = r_dist
                neighbor[r] = r_neighbor

            # select target regions
            r_id = np.argmin(dist)

            # update seed
            seed_region[r_id].add(neighbor[r_id])

            # update other seed region's neighbor
            for r in np.count_nonzero(r_index):
                seed_region[r].remove_neghbor(neighbor[r])

        return seed_region









