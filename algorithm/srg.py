# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""
An object to represent the region and its associated attributes

"""
import numpy as np


class Region(object):
    def __init__(self, vox_pos, vox_feat, r_id, neighbor):
        """

        Parameters
        ----------
        vox_pos: coordinates array, n_voxel x 3 np.array
        vox_feat : feature array, n_voxel x n_feature np.array
        r_id_: region id, a unique scalar
        neighbor: neighbor region list

        Returns
        -------

        """
        self.vox_pos = vox_pos
        self.vox_feat = vox_feat
        self.id = r_id
        self.feat = np.mean(vox_feat)

        # component regions list
        self.component = []
        self.component.append(self)

        # neighbor region list
        self.neighbor = neighbor

    def merge(self, region):
        # merge pos and feat
        self.vox_pos = np.append(self.vox_pos, region.vox_pos)
        self.vox_feat = np.append(self.vox_feat, region.vox_feat)

        # add region to the component
        self.component.append(region)

        # add region's neighbor to the seed's neighbor
        for i in range(len(region.neighbor)):
            self.add_neighbor(region.neighbor[i])

    def add_neighbor(self, region):
        if region not in self.component and region not in self.neighbor:
            self.neighbor.append(region)

    def remove_neighbor(self, region):
        if region in self.neighbor:
            self.neighbor.remove(region)

    def size(self):
        return self.vox_pos.shape[0]

    def sum_feat(self):
        self.feat = np.mean(self.vox_feat)
        return self.feat

    def nearest_neighbor(self):
        feat = [region.sum_feat() for region in self.neighbor]
        dist = np.absolute(np.array(feat) - self.sum_feat())
        index = np.argmin(dist)

        return self.neighbor[index], dist[index]


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
        """

        Parameters
        ----------
        meth : method to be used to segment the image

        Returns
        -------

        """
        self.meth = meth

    def compute(self, image, n_region, mask=None):
        """

        Parameters
        ----------
        image: image array to be represented
        n_region : number of region to be generate
        mask: mask image to give region of interest

        Returns
        -------
        regions : a list of region

        """

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
        """

        Parameters
        ----------
        similarity_measure
        stop_criteria

        Returns
        -------

        """
        self.similarity_measure = similarity_measure
        self.stop_criteria = stop_criteria

    def compute(self, seed_region):
        """

        Parameters
        ----------
        seed_region: a list for seed regions

        Returns
        -------

        """

        n_seed = len(seed_region)
        region_size = np.zeros(n_seed)
        for r in range(n_seed):
            region_size[r] = seed_region[r].size()

        dist = np.empty(n_seed)
        dist.fill(np.inf)
        while np.any(np.less(region_size < self.stop_criteria)):
            r_to_grow = np.less(region_size < self.stop_criteria)
            dist[np.logical_not(r_to_grow)] = np.inf
            neighbor = []
            r_index = np.nonzero(r_to_grow)[0]

            for i in np.arange(r_index.shape[0]):
                # find the nearest neighbor for the each seed region
                r_neighbor, r_dist, = seed_region[r_index[i]].nearest_neighbor()
                dist[i] = r_dist
                neighbor.append(r_neighbor)

            # find the seed which has min neighbor in this iteration
            r = np.argmin(dist)

            # merge the neighbor to the seed
            seed_region[r_index[r]].merge(neighbor[r])

            # remove the neighbor from the neighbor list of all seeds
            for i in r_index:
                seed_region[i].remove_neghbor(neighbor[r])

        return seed_region









