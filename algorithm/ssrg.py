from ..core.dataobject import Hemisphere
import numpy as np
import time


class Region(object):
    """
    An object to represent the region and its associated attributes
    """

    def __init__(self, r_id, vtx_feat):
        """
        Parameters
        ----------
        r_id: int
            region id, a unique scalar(actually is vertex number)
        vtx_feat : dict
            key is the id, value is a numpy array with shape (n_features,)

        Returns
        -------
        """

        # Initialize fields
        # ------------------------------------------------------------------------------------
        self.id = r_id
        self.vtx_feat = vtx_feat
        self.mean_feat = self._mean_feat()

        # component regions list
        self.component = []  # may be removed at a new version
        self.component.append(self)

        # neighbor region list
        self.neighbor = []

    def _mean_feat(self):
        """
        calculate the mean of the len(self.vtx_feat) feature arrays with the shape of (n_features,)
        """

        # initialize the sum
        s = None
        for v in self.vtx_feat.values():
            s = v.copy()
            s.fill(0)
            break

        # calculate the sum
        for v in self.vtx_feat.values():
            s += v

        return s/len(self.vtx_feat)

    def merge(self, region):
        """
        merge the region to self

        Parameters
        ---------
        region: Region
            a instance of the class Region

        Returns
        -------
        """

        # merge features
        self.vtx_feat.update(region.vtx_feat)

        # add region to the component
        self.component.append(region)

        # add region's neighbor to the seed's neighbor
        for i in range(len(region.neighbor)):
            self.add_neighbor(region.neighbor[i])

    def add_neighbor(self, region):
        """
        add the neighbor for self

        Parameters
        ----------
        region: Region
            a instance of the class Region

        Returns
        -------
        """

        if region not in self.component and region not in self.neighbor:
            self.neighbor.append(region)

    def remove_neighbor(self, region):
        """
        remove the neighbor for self

        Parameters
        ----------
        region: Region
            a instance of the class Region

        Returns
        -------
        """

        if region in self.neighbor:
            self.neighbor.remove(region)

    def size(self):
        """
        the size of self

        Returns
        -------
            the size of self
        """

        return len(self.component)

    def r_measure(self):
        """
        calculate the measure used to determine the distance with neighbor

        Returns
        -------
            the measure of self
        """

        measure = np.mean(self.mean_feat)
        return measure

    def nearest_neighbor(self):
        """
        find the nearest neighbor of self

        Returns
        -------
            the nearest neighbor and its distance corresponding to self
        """

        measure = [region.r_measure() for region in self.neighbor]
        dist = np.absolute(np.array(measure) - self.r_measure())
        index = np.argmin(dist)

        return self.neighbor[index], dist[index]


class SurfaceToRegions(object):

    def __init__(self, hemisphere):
        """
        represent the surface to preliminary regions

        Parameters
        ----------
        hemisphere: Hemisphere
            a instance of the class Hemisphere

        Returns
        -------
            a instance of itself
        """

        if not isinstance(hemisphere, Hemisphere):
            raise TypeError("The argument hemisphere must be a instance of Hemisphere!")

        geo = hemisphere.geo
        scalars = hemisphere.scalar_dict

        # just temporarily use the field to find suitable seed_region
        # self.scalar = scalars.values()[0]

        self.regions = []
        for r_id in range(len(geo.x)):

            vtx_feat = dict()
            vtx_feat[r_id] = np.zeros(len(scalars))
            for i, key in enumerate(hemisphere.scalar_order):
                vtx_feat[r_id][i] = scalars[key].scalar_data[r_id]

            self.regions.append(Region(r_id, vtx_feat))

        # find neighbors' id for each region
        # list_of_neighbor_set = [set()] * len(self.regions)  # each element is the reference to the same set object
        list_of_neighbor_set = [set() for i in range(len(self.regions))]
        f = geo.faces
        for face in f:
            for vtx_id in face:
                list_of_neighbor_set[vtx_id].update(set(face))

        # add neighbors
        for r_id in range(len(self.regions)):
            list_of_neighbor_set[r_id].remove(r_id)
            for neighbor_id in list_of_neighbor_set[r_id]:
                self.regions[r_id].add_neighbor(self.regions[neighbor_id])

    def get_regions(self):
        return self.regions

    def get_seed_region(self):
        """
        just temporarily use the method to find suitable seed_region

        return
        ------
            the list of seeds
        """

        seed_region = []

        data = self.scalar.scalar_data
        max = data.max()
        seed_index = np.nonzero(data == max)[0]

        cnt = 0
        for index in seed_index:

            cnt += 1
            if cnt > 10:
                break

            seed_region.append(self.regions[index])

        return seed_region


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
    _compute(region,image)
        do region growing
    """

    def __init__(self, seed_region, stop_criteria=1000, similarity_measure=None):
        """
        Parameters
        ----------
        similarity_measure:
        stop_criteria:

        Returns
        -------
        """

        # initialize the fields
        self.similarity_measure = similarity_measure
        self.stop_criteria = stop_criteria
        self.seed_region = seed_region

        # call methods of the class
        self._compute()

    def _compute(self):
        """
        do region growing
        """

        n_seed = len(self.seed_region)
        region_size = np.zeros(n_seed)
        for r in range(n_seed):
            region_size[r] = self.seed_region[r].size()

        dist = np.empty(n_seed)
        dist.fill(np.inf)  # fill with 'Inf'(infinite), similar to 'NaN'
        # Not a Number (NaN), positive infinity and negative infinity evaluate to
        # True because these are not equal to zero.
        neighbor = [None] * n_seed

        while np.any(np.less(region_size, self.stop_criteria)):
            r_to_grow = np.less(region_size, self.stop_criteria)
            dist[np.logical_not(r_to_grow)] = np.inf

            r_index = np.nonzero(r_to_grow)[0]

            for i in np.arange(r_index.shape[0]):
                # find the nearest neighbor for the each seed region
                r_neighbor, r_dist, = self.seed_region[r_index[i]].nearest_neighbor()
                dist[r_index[i]] = r_dist
                neighbor[r_index[i]] = r_neighbor

            # find the seed which has min neighbor in this iteration
            r = np.argmin(dist)

            # merge the neighbor to the seed
            self.seed_region[r].merge(neighbor[r])

            # update region_size
            region_size[r] = region_size[r] + neighbor[r].size()

            # remove the neighbor from the neighbor list of all seeds
            for i in r_index:
                self.seed_region[i].remove_neighbor(neighbor[r])

    def region2text(self):
        """
        save region into text
        """

        for seed in self.seed_region:

            vtx_feat = seed.vtx_feat
            X = np.zeros(len(vtx_feat))
            for index, key in enumerate(vtx_feat.keys()):
                X[index] = key

            file_name = "/nfs/j3/userhome/chenxiayu/workingdir/" + str(seed.id) + "_srg_zstat.label"
            header = str("the number of vertex: " + str(len(vtx_feat)))
            np.savetxt(file_name, X, fmt='%d',
                       header=header, comments="# ascii, label vertexes saved by genius cxy!\n")