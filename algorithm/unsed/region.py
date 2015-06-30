# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""
An object to represent the region and its associated attributes

"""

from algorithm.unsed.neighbor import *

class Region(object):
    """
    An object to represent the region and its associated attributes
    Attributes
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
        seed: list of tuple coordinates [(x1,y1,z1),(x2,y2,z2),]
            The element of the list is a series tuples, each of which in turn holds the coordinates of a point
        neighbor_element: SpatialNeighbor object
            The neighbor generator which generate the spatial neighbor(coordinates)for a point
        label: numpy 2d array
            The coordinates of the points which have been merged into the regions
        neighbor: numpy 2 array
        The coordinates of the points which is the neighbor of the merged region

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
            self.sampling_seed = self.seed[np.random.choice(self.seed.shape[0], sampling_num, replace=True), :]
        else:
            self.sampling_seed = self.seed
        return self.sampling_seed

    def get_sampling_seed(self):
        """
        Return the smapling seed.

        """
        return self.sampling_seed

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
        neighbor: numpy 2d array
            Each row represents the coordinates for a pixels
        """
        self.label = np.append(self.label, label, axis=0)

    def set_label(self, label):
        """
        Set the coordinates of new label to the label of region.
        Parameters
        neighbor: numpy 2d array
            Each row represents the coordinates for a pixels
        """
        self.label = label

    def get_label(self):
        """
        Get the the coordinates of the labeled pixels
        """
        return self.label

    def get_neighbor(self):
        """
        Get the the coordinates of the neighbor pixels
        """
        if self.neighbor == None:
            self.add_neighbor(self.label)
        return self.neighbor

    def set_neighbor(self, neighbor):
        """
        Set the the coordinates of the neighbor pixels
        """
        self.neighbor = neighbor

    def add_neighbor(self, label):
        """
        Add the coordinates of new neighbor to the neighbor of region.
        Parameters
        neighbor: numpy 2d array
        Each row represents the coordinates for  a pixels
        """
        neighbor = self.neighbor_element.compute(label)
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

class SlicRegion(Region):
    """
    An object to represent the supervoxel region and its associated attributes
    Attributes
    seed: list of tuple supervoxel values
        The element of the list is a series integer values, each of which in turn holds the value of a supervoxel
    label: list of labled supervoxel values
        The values of the supervoxel points which have been merged into the regions
    neighbor: list of neighbored supervoxel values
        The values of the neighbors of certain supervoxel point which have been merged into the regions

    """
    def __init__(self, seed, slic_image):
        """
        Parameters
        seed: list of tuple supervoxel values
            The element of the list is a series integer values, each of which in turn holds the value of a supervoxel
        label: list of labled supervoxel values
            The values of the supervoxel points which have been merged into the regions
        neighbor: list of neighbored supervoxel values
            The values of the neighbors of certain supervoxel point which have been merged into the regions

        """
        if isinstance(seed, np.ndarray):
            seed = seed.tolist()
        self.seed = seed
        self.slic_image = slic_image
        self.label = self.seed
        self.neighbor = []
        self.add_neighbor(seed)

    def label_size(self):
        """
        Return the size of current label.
        """
        return len(self.label)

    def neighbor_size(self):
        """
        Return the size of current neighbor.
        """
        return len(self.neighbor)

    def add_label(self, label):
        if label not in self.label and label in self.neighbor:
            self.label.append(label)

    def set_label(self, label):
        """
        Set the coordinates of new label to the label of region.
        Parameters
        neighbor: numpy 2d array
            Each row represents the coordinates for a pixels
        """

        if not isinstance(label, list):
            label = [label]
        self.labels = label


    def get_label_vaue(self):
        """
        Get the the coordinates of the labeled pixels
        """
        return self.label

    def get_label(self):
        region_image = np.zeros_like(self.slic_image)
        for label in self.label:
            region_image[self.slic_image == label] = 1

        return np.array(np.nonzero(region_image)).T

    def add_neighbor(self, *args):
        #only compute for 3D or 2D image
        def add_neighbor_basic(self, label):
            from scipy.ndimage.morphology import binary_dilation
            #compute new neighbors
            neighbor_slic = binary_dilation(self.slic_image == label)
            neighbor_slic[self.slic_image == label] = 0
            neighbor_values = np.unique(self.slic_image[neighbor_slic > 0])
            neighbor_values = np.delete(neighbor_values, 0)

            for neighbor_value in neighbor_values:
                if neighbor_value not in self.label and neighbor_value not in self.neighbor:
                    self.neighbor.append(neighbor_value)

        def add_neighbor_slic(self, neighbor, similarity_criteria, label_num):
            """
            Add the coordinates of new neighbor to the neighbor of region.
            Parameters
            neighbor: numpy 2d array
                Each row represents the coordinates for  a pixels
            SSL: numpy 2d array
            """
            neighbor = self.neighbor_element.compute(neighbor)
            # find the neighbor which have been in neighbor or in label list
            marked = np.logical_or(utils.in2d(neighbor, self.neighbor),
                                   utils.in2d(neighbor, self.label))
            # delete the marked neighbor
            neighbor = np.delete(neighbor, np.nonzero(marked), axis=0)
            # Add unmarked neighbor to the region neighbor and update the neighbor size
            ssl = similarity_criteria.get_ssl()
            boundary = similarity_criteria.get_boundary()
            neighbor_labeled = []
            neighbor_unlabeled = []
            for element in neighbor:
                if element in ssl.keys():
                    neighbor_labeled.append(ssl[element])
                elif element not in ssl.keys() and element not in boundary:
                    neighbor_unlabeled.append(element)

            neighbor_labeled_values = neighbor_labeled.values()
            for i in range(len(1, neighbor_labeled_values)):
                if neighbor_labeled_values[0] != neighbor_labeled_values[i]:
                    similarity_criteria.add_boundary_element(neighbor)
                    return

            for i in range(len(neighbor_unlabeled)):
                self.neighbor = np.append(self.neighbor, neighbor_unlabeled[i], axis=0)
                similarity_criteria.add_ssl_element(neighbor_unlabeled[i], label_num)
        if len(args) == 1:
            return add_neighbor_basic(self, *args)
        else:
            return add_neighbor_slic(self, *args)

    def remove_neighbor(self, neighbor):
        if neighbor in self.neighbor:
            self.neighbor.remove(neighbor)

    def get_neighbor_value(self):
        """
        Get the coordinates of the labeled pixels
        """
        return self.neighbor

    def get_neighbor(self):
        region_image = np.zeros_like(self.slic_image)
        for neighbor in self.neighbor:
            region_image[self.slic_image == neighbor] = 1
        return np.array(np.nonzero(region_image)).T

    def get_slic_image(self):
        return self.slic_image

    def compute_boundary(self):

        """
            Compute the  boundary for the label
        """
        return self.neighbor


