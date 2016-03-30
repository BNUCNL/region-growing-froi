# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
from skimage import segmentation, filters
from skimage.future import graph


class Region(object):

    """
    An object to represent the region and its associated attributes
    """
    def __init__(self, vox_pos, vox_feat, r_id):
        """
        Parameters
        ----------
        vox_pos: coordinates array, n_voxel x 3 np.array
        vox_feat : feature array, n_voxel x n_feature(such as gray value) np.array
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
        self.neighbor = []

    def merge(self, region):
        # merge pos and feat
        self.vox_pos = np.concatenate((self.vox_pos, region.vox_pos))  # or vstack((a, b))
        self.vox_feat = np.concatenate((self.vox_feat, region.vox_feat))

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
    image: numpy ndarray
        the image you want to segment
    meth: original or divided
    mask: mask to indicate the spatial extent to be represented
    Returns
    -------
    regions: a list of Region objects
    """
    def __init__(self, meth='slic'):
        """
        Parameters
        ----------
        meth : str
            method to be used to segment the image
        Returns
        -------
        """

        self.meth = meth
        self.image_shape = None
        self.regions = None

    def compute(self, image, n_region, mask=None):
        """
        Parameters
        ----------
        image: numpy ndarray
            image array to be represented as regions
        n_region : int
            the number of regions to be generated
        mask: mask image to give region of interest
        Returns
        -------
        regions : a list of regions
        """

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
        # edge_map = filters.sobel(gray_image)  # just for 2-D image
        edge_map = filters.laplace(gray_image)

        # Given an image's initial segmentation and its edge map this method constructs the
        # corresponding Region Adjacency Graph (RAG). Each node in the RAG represents a set of
        # pixels within the image with the same label in labels. The weight between two adjacent
        # regions is the average value in edge_map along their boundary.
        rag = graph.rag_boundary(labels, edge_map)

        regions = []
        n_labels = labels.max() + 1
        for r in np.arange(n_labels):
            # get vox_pos
            position = np.transpose(np.nonzero(labels == r))

            # get vox_feat
            vox_feat = np.zeros((position.shape[0], 1))
            n_D = position.shape[1]
            for i in range(position.shape[0]):
                if n_D == 2:
                    vox_feat[i][0] = image[position[i][0], position[i][1]]
                elif n_D == 3:
                    vox_feat[i][0] = image[position[i][0], position[i][1], position[i][2]]
                else:
                    raise RuntimeError("We just consider 2_D and 3_D images at present!")

            regions.append(Region(position, vox_feat=vox_feat, r_id=r))

        for r in range(n_labels):
            for r_key in rag.edge[r].keys():
                regions[r].add_neighbor(regions[r_key])

        self.regions = regions
        self.image_shape = image.shape
        return regions

    def regions2image(self):
        """
        transform regions to the image

        return
        ------
        ndarray
        """

        if self.image_shape is None or self.regions is None:
            raise ValueError("Please call the method 'compute()' previously!")

        img = np.zeros(self.image_shape)
        n_D = len(self.image_shape)

        for region in self.regions:
            for i in range(region.vox_pos.shape[0]):
                coord = region.vox_pos[i]
                if n_D == 2:
                    img[coord[0], coord[1]] = region.vox_feat[i][0]
                elif n_D == 3:
                    img[coord[0], coord[1], coord[2]] = region.vox_feat[i][0]
                else:
                    raise RuntimeError("We just consider 2_D and 3_D images at present!")

        return img

    def regions2labels(self):
        """
        transform regions to the image of labels

        return
        ------
        ndarray
        """

        if self.image_shape is None or self.regions is None:
            raise ValueError("Please call the method 'compute()' previously!")

        label_img = np.zeros(self.image_shape)
        n_D = len(self.image_shape)

        for region in self.regions:
            for i in range(region.vox_pos.shape[0]):
                coord = region.vox_pos[i]
                if n_D == 2:
                    label_img[coord[0], coord[1]] = region.id
                elif n_D == 3:
                    label_img[coord[0], coord[1], coord[2]] = region.id
                else:
                    raise RuntimeError("We just consider 2_D and 3_D images at present!")

        return label_img


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
    def __init__(self, stop_criteria):
        """
        Parameters
        ----------
        similarity_measure
        stop_criteria
        Returns
        -------
        """
        # self.similarity_measure = similarity_measure
        self.stop_criteria = stop_criteria
        self.seed_region = None

    def set_seed(self, x, y):
        pass

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
        dist.fill(np.inf)  # fill with 'Inf'(infinite), similar to 'NaN'
        # Not a Number (NaN), positive infinity and negative infinity evaluate to
        # True because these are not equal to zero.
        neighbor = [None for i in range(n_seed)]

        while np.any(np.less(region_size, self.stop_criteria)):
            r_to_grow = np.less(region_size, self.stop_criteria)
            dist[np.logical_not(r_to_grow)] = np.inf

            r_index = np.nonzero(r_to_grow)[0]

            for i in np.arange(r_index.shape[0]):
                # find the nearest neighbor for the each seed region
                r_neighbor, r_dist, = seed_region[r_index[i]].nearest_neighbor()
                dist[r_index[i]] = r_dist
                neighbor[r_index[i]] = r_neighbor

            # find the seed which has min neighbor in this iteration
            r = np.argmin(dist)

            # merge the neighbor to the seed
            seed_region[r].merge(neighbor[r])

            # update region_size
            region_size[r] = region_size[r] + neighbor[r].size()

            # remove the neighbor from the neighbor list of all seeds
            # (I think it should be all regions rather than all seeds)
            for i in r_index:
                seed_region[i].remove_neighbor(neighbor[r])

        self.seed_region = seed_region

        return seed_region

    def region2image(self, image_shape):
        """
        transform region into image
        Parameters
        ----------
        image_shape : instance.image_shape of SeededRegionGrowing

        Returns
        -------
        img : matrix of image

        """

        img = np.zeros(image_shape)
        n_D = len(image_shape)
        for region in self.seed_region:
            for i in range(region.vox_pos.shape[0]):
                coord = region.vox_pos[i]
                if n_D == 2:
                    img[coord[0], coord[1]] = region.vox_feat[i][0]
                elif n_D == 3:
                    img[coord[0], coord[1], coord[2]] = region.vox_feat[i][0]
                else:
                    raise RuntimeError("We just consider 2_D and 3_D images at present!")

        return img

if __name__ == "__main__":

    from skimage import data
    import matplotlib.pyplot as plt
    from skimage.color import rgb2gray

    image = rgb2gray(data.hubble_deep_field()[0:500, 0:500])

    ritr = RepresentImageToRegion()
    regions = ritr.compute(image, 1000)
    img = ritr.regions2image()
    label_img = ritr.regions2labels()

    seed_region = [regions[100], regions[800]]
    srg = SeededRegionGrowing(5000)
    srg.compute(seed_region)
    srg_img = srg.region2image(ritr.image_shape)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    ax1.imshow(image)
    ax2.imshow(img)
    ax3.imshow(label_img)
    ax4.imshow(srg_img)

    plt.show()
