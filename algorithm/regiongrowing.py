import random

import numpy as np
from scipy.spatial import distance

import utils


class SeededRegionGrowing:
    """
    Seeded region growing with a fixed threshold.
    """

    def __init__(self, image, seeds, similarity_criteria, stop_criteria, neighbor):
        """
        Parameters
        -----------------------------------------------------
<<<<<<< HEAD
        target_image: input image, a 2D/3D/4D Nifti1Image object
        seeds: a set of coordinates or a region mask
        value: stop threshold.
        """
        if isinstance(target_image, nib.nifti1.Nifti1Image):
            self.target_image = target_image.get_data()
            if len(target_image.shape) > 4 or len(target_image.shape) < 2:
                raise ValueError("Target image must be a 2D/3D/4D or Nifti1Image format file.")
        elif isinstance(target_image, np.ndarray):
            self.target_image = target_image
            if len(target_image.shape) > 4 or len(target_image.shape) < 2:
                raise ValueError("Target image must be a 2D/3D/4D or Nifti1Image format file.")
=======
        image: input image, a 2D/3D/4D array
        seeds: an instance of class Seed
        similarity_criteria: an instance of class SimilarityCriteria
        stop_criteria: an instance of class StopCriteria
        neighbor: an instance of class SpatialNeighbor
        """

        if 2 <= len(image.shape) <= 4:
            self.image = image
        else:
            raise ValueError("Target image must be a 2D/3D/4D array.")
>>>>>>> fa66cac6a7943905c31b7719abb5f7733a2c546d

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
        growing  a region with specified similarity and stop criterion

        """

        while not self.stop_criteria.isstop():
            nearest_neighbor = self.similarity_criteria.compute(self.image, self.region)

            self.region.add_label(nearest_neighbor)

            self.region.remove_neighbor(nearest_neighbor)

            self.region.add_neighbor(self.neighbor.compute(nearest_neighbor))

            self.stop_criteria.compute(self.image, self.region)
        return self.region


class Seeds(object):
    """
    Seeds.
    """

    def __init__(self, coords):
        """
        Parameters
        -----------------------------------------------------
        coords: tuple of tuple(((x1,y1,z1),(x2,y2,z2)),()). The out tuple indicates the number of group seeds, the inner
          tuple indicates the number of voxels of each group of seeds

        """

        self.coords = coords

    def generate(self):
        """
        Generating new seeds.       """
        return self.coords


class RandomSeeds(Seeds):
    """
    Random Seeds.
    """

    def __init__(self, coords, sampling_number=10):
        """
        Init seeds.
        Parameters
        -----------------------------------------------------
        value: a set of coordinates or a region mask.
        """
        super(RandomSeeds, self).__init__(coords)

        if not isinstance(sampling_number, int):
            raise ValueError("The random_number must be int type.")
        else:
            self.sampling_number = sampling_number

    def generate(self):
        """
        Generating new seeds.

        return: sampling_coords: sample multiple times and sample a voxel from each group of seeds

        """
        sampling_coords = []
        for r in range(self.sampling_number):
            single_sampling = []
            for g in range(len(self.coords)):
                single_sampling.append(random.choice(self.coords[g]))
            sampling_coords.append(single_sampling)

        return sampling_coords


class SimilarityCriteria:
    """
    Similarity criteria: compute the similarity between region and neighbor voxels
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

    def compute(self, image, region, mask_image=None, prior_image=None):
        """
        Compute the similarity between the labeled region and neighbors.

        Parameters
        -----------------------------------------------------
        region: the labeled region, an instance of class Region.
        image: The raw image where the region grow from.
        mask_image: the mask image may be used in the compute process. which should be a ndarray.
        prior_image:the prior image may be used in the compute process. which should be a ndarray.
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
    Stop criteria.
    """
<<<<<<< HEAD
    def __init__(self, name='region_size', threshold=None, step=None):
=======

    def __init__(self, threshold=None, criteria_metric='size'):
>>>>>>> fa66cac6a7943905c31b7719abb5f7733a2c546d
        """
        Parameters
        -----------------------------------------------------
        name:'region_homogeneity','region_morphology','region_difference', 'region_size', default is 'region_difference'
        threshold: a int value or None, default is None which means the adaptive method will be used.
        """

        self.metric = criteria_metric
        self.threshold = threshold
<<<<<<< HEAD
        self.setp = step
=======
        self.stop = False
>>>>>>> fa66cac6a7943905c31b7719abb5f7733a2c546d

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

    def set_threshold(self, threshold):
        """
        Get the threshold of the stop criteria..
        """
        self.threshold = threshold

    def get_threshold(self):
        """
        Get the threshold of the stop criteria..
        """
        return self.threshold

    def set_step(self, step):
        """
        Get the step of the stop criteria..
        """
        self.step = step

    def get_step(self):
        """
        Get the step of the stop criteria..
        """
        return self.step

    def compute(self, image, region, mask_image=None, prior_image=None):
        """

        check whether growing should stop according to the region and image

        """

        if self.metric == 'size':
            if region.label_size > self.threshold:
                self.stop = True

    def isstop(self):
        return self.stop


class Region(object):
    """
    Region
    """

    def __init__(self, region_label, region_neighbor):
        """
        Parameters
        -----------------------------------------------------
        region_extent and region_neighbor is Nx3 array which keep the coordinates of the voxels
        """

        if not isinstance(region_label, np.ndarray):
            raise ValueError("The current region of the Region class must be ndarray type. ")

        if not isinstance(region_neighbor, np.ndarray):
            raise ValueError("The neighbor of the Region class must be ndarray type. ")

        buffer_size = 10000
        self.label = np.zeros((buffer_size, 3), dtype=int)
        self.neighbor = np.zeros((buffer_size, 3), dtype=int)

        self.label_size = region_label.shape[0]
        self.label[:self.label_size, :] = region_label

        self.neighbor_size = region_neighbor.shape[0]
        self.neighbor[:self.neighbor_size, :] = region_neighbor


    def set_label(self, region_label):
        """
        Get the neighbor.
        """
        self.label = region_label

    def get_label(self):
        """
        Get the neighbor.
        """
        return self.label

    def set_neighbor(self, region_neighbor):
        """
        Set the region neighbor.
        """
        self.neighbor = region_neighbor

    def get_neighbor(self):
        """
        Get the  region neighbor.
        """
        return self.neighbor

    def add_label(self, new_label):

        self.label[self.label_size, :] = new_label
        self.label_size += 1

    def add_neighbor(self, new_neighbor):

        # print utils.in2d(new_neighbor, self.neighbor[:self.neighbor_size, :])
        marked = np.logical_or(utils.in2d(new_neighbor, self.neighbor[:self.neighbor_size, :]),
                               utils.in2d(new_neighbor, self.label[:self.label_size, :]))

        new_neighbor = np.delete(new_neighbor, np.nonzero(marked), axis=0)

        # print new_neighbor

        self.neighbor[self.neighbor_size:(self.neighbor_size + new_neighbor.shape[0]), :] = new_neighbor
        self.neighbor_size = self.neighbor_size + new_neighbor.shape[0]

    def remove_neighbor(self, new_label):

        idx = np.nonzero(utils.in2d(self.neighbor[:self.neighbor_size, :], new_label))[0]
        self.neighbor = np.delete(self.neighbor, idx, 0)
        self.neighbor_size -= len(idx)


def compute_inner_boundary(self):
    """
        Compute the inner boundary
<<<<<<< HEAD
        """
        inner_b = []
        # for i in inner_region_cor:
        #     if self.is_neiflag(flag_image, i, 1):
        #         if inner_b == []:
        #             inner_b = i
        #         else:
        #             inner_b= np.vstack((inner_b, i))
        return np.array(inner_b)
=======
    """
    #Do something here.
>>>>>>> fa66cac6a7943905c31b7719abb5f7733a2c546d

    pass


def compute_external_boundary(self):
    """
        Compute the external boundary
    """
    #Do something here.
    pass


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
<<<<<<< HEAD
        #Do something here.
        return self.get_region_neighbor()
=======
>>>>>>> fa66cac6a7943905c31b7719abb5f7733a2c546d


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
<<<<<<< HEAD
    def __init__(self, target_image, seeds, similarity_criteria, stop_criteria, connectivity, region_sequence=None):
        if not isinstance(seeds, np.ndarray):
            seeds = np.array(seeds)

        SeededRegionGrowing.__init__(self, target_image, seeds, similarity_criteria, stop_criteria, connectivity, region_sequence=None)
=======
>>>>>>> fa66cac6a7943905c31b7719abb5f7733a2c546d

    def __init__(self, image, seeds, similarity_criteria, stop_criteria, neighbor, optimizer):
        super(AdaptiveSRG, self).__init__(image, seeds, similarity_criteria, stop_criteria, neighbor)
        self.optimizer = optimizer


    def grow(self):
        """
        Adaptive region growing.
        """
<<<<<<< HEAD
        region_sequence = []
        thresholds = self.stop_criteria.computing()
        for threshold in thresholds:
            self.get_stop_criteria().set_threshold(threshold)
            self.get_stop_criteria().set_step(None)
            region_sequence.append(self.grow())
        return PCB(region_sequence, self.target_image)



class RegionOptimizer():
    """
    Region optimizer.
    """
    def __init__(self, name="Region optimizer"):
        """
        Region optimizer init.
        """
        self.name = name

    def optimize(self, region_sequence, raw_image, mask_image=None, prior=None):
        """
        Parameters
        -----------------------------------------------------
        region_sequence: region sequence
        opt_measurement: the optimize measurement.
        mask_image: the mask image may be used in the compute process. which should be a ndarray type.
        prior_image:the prior image may be used in the compute process. which should be a ndarray type.
        """
        return self.optimize(self, region_sequence, raw_image, mask_image=None, prior=None)


class PCB(RegionOptimizer):
    """
    PCB region optimizer.
    """
    def __init__(self, name="PCB region optimizer"):
        """
        PCB region optimizer init.
        """
        self.name = name

    def optimize(self, region_sequence, raw_image, mask_image=None, prior=None):
        """
        Parameters
        -----------------------------------------------------
        region_sequence: region sequence
        opt_measurement: the optimize measurement.
        mask_image: the mask image may be used in the compute process. which should be a ndarray type.
        prior_image:the prior image may be used in the compute process. which should be a ndarray type.
        """
        for region in region_sequence:
            outer_boundary = region.compute_EB()
            inner_boundary = region.compute_IB()
            contrast = contrast + [np.mean(inner_boundary_val) - np.mean(outer_boundary)]

        return optimizer_region


class ACB(RegionOptimizer):
    """
    ACB region optimizer.
    """
    def __init__(self, name="ACB region optimizer"):
        """
        ACB region optimizer init.
        """
        self.name = name

    def optimize(self, region_sequence, raw_image, mask_image=None, prior=None):
        """
        Parameters
        -----------------------------------------------------
        region_sequence: region sequence
        opt_measurement: the optimize measurement.
        mask_image: the mask image may be used in the compute process. which should be a ndarray type.
        prior_image:the prior image may be used in the compute process. which should be a ndarray type.
        """
        for region in region_sequence:
            outer_boundary = region.compute_EB()
            inner_boundary = region.compute_IB()
            contrast = contrast + [np.mean(np.array(inner_list)) - np.mean(out_boundary)]

        return optimizer_region
=======
>>>>>>> fa66cac6a7943905c31b7719abb5f7733a2c546d


if __name__ == "__main__":
    seed_coors = (((1, 2, 3), (3, 2, 1)), ((4, 5, 6), (6, 5, 1)))
    seeds3d = Seeds(seed_coors)
    print seeds3d.generating()

    randseeds = RandomSeeds(seed_coors)
    print randseeds.generating()


    #similarity_criteria = NeighborSimilarity(metric='euclidean',)
    #stop_criteria = StopCriteria(name='region_size', threshold=300)
    #connectivity = Connectivity('6')

