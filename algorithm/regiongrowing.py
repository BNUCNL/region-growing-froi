import random

import numpy as np
import nibabel as nib
from scipy.spatial import distance


class SeededRegionGrowing:
    """
    Seeded region growing with a fixed threshold.
    """

    def __init__(self, target_image, seeds, similarity_criteria, stop_criteria, neighbor):
        """
        Parameters
        -----------------------------------------------------
        target_image: input image, a 2D/3D/4D Nifti1Image object or N-D array
        seeds: an instance of class Seed
        similarity_criteria: an instance of class SimilarityCriteria
        stop_criteria: an instance of class StopCriteria
        neighbor: an instance of class Connectivity(SpatialNeighbor)
        """

        if isinstance(target_image, nib.nifti1.Nifti1Image):
            target_image = target_image.get_data()

        if 2 <= len(target_image.shape) <= 4:
            self.image = target_image
        else:
            raise ValueError("Target image must be a 2D/3D/4D.")

        self.seeds = seeds
        self.similarity_criteria = similarity_criteria
        self.stop_criteria = stop_criteria
        self.neighbor = neighbor

        region_label = self.seeds
        region_neighbor = []
        for i in range(len(self.seeds.coords)):
            for j in range(len(self.seeds.coords[i])):
                region_neighbor = self.neighbor(self.seeds.coords)

        self.region = Region(region_label, region_neighbor)


    def set_image(self, target_image):
        self.image = target_image

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
        self.set_similarity_criteria(similarity_criteria)

    def get_similarity_criteria(self):
        """
        Get the similarity criteria.
        """
        return self.get_similarity_criteria()

    def set_region(self, region):
        """
        Set the region sequence.
        """
        self.region = region

    def get_region_sequence(self):
        """
        Get the region sequence..
        """
        return self.region

    def grow(self):
        """
        growing  a region with specified similarity and stop criterion

        """

        while not self.stop_criteria.isstop():
            nearest_neighbor = self.similarity_criteria.computing(self.region, self.image)
            self.region.add_label(nearest_neighbor)
            self.neighbor.add_neighbor(self.neighbor(nearest_neighbor))

            self.stop_criteria.computing(self.region, self.image)
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

    def generating(self):
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

    def generating(self):
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
    Similarity criteria: compute the similairity between region and neighbor voxels
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

    def computing(self, region, img=None, mask_img=None, prior_img=None):
        """
        Compute the  similarity.
        Parameters
        -----------------------------------------------------
        region:The region growing.
        raw_image: The raw image.
        mask_image: the mask image may be used in the compute process. which should be a ndarray type.
        prior_image:the prior image may be used in the compute process. which should be a ndarray type.
        """

        region_mean = np.mean(img[region.extent, :])
        neighbor_intensity = img(region.neighbor)
        dist = distance.cdist(region_mean, neighbor_intensity, self.metric)
        index = dist.argmin()


class StopCriteria(object):
    """
    Stop criteria.
    """

    def __init__(self, target_image, region, criterion_type='morphlogy', criterion_metric='region_size',
                 threshold=None):
        """
        Parameters
        -----------------------------------------------------
        name:'region_homogeneity','region_morphology','region_difference', 'region_size', default is 'region_difference'
        threshold: a int value or None, default is None which means the adaptive method will be used.
        """

        self.type = criterion_type
        self.name = criterion_metric
        self.threshold = threshold
        self.stop = False

    def set_metric(self, metric_name):
        """
        Get the name of the stop criteria..
        """
        self.metric = metric_name

    def get_metric(self):
        """
        Get the name of the stop criteria..
        """
        return self.metric


    def computing(self, region, raw_image=None, mask_image=None, prior_image=None):
        """
        Set the stop criteria.
        """
        if cmp(self.type, 'morphology'):
            if cmp(self.metric, 'region_size'):
                region_size = np.count_nonzero(region.extent)
                if region_size >= self.threshold:
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
        self.label = np.zeros((buffer_size, 4))
        self.neighbor = np.zeros(buffer_size, 4)

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

        self.label[self.label_size:(self.label_size + new_label.shape[0]), :] = new_label
        self.label_size = self.label_size + new_label.shape[0]

    def add_neighbor(self, new_neighbor):

        self.label[self.neighbor_size:(self.neighbor_size + new_neighbor.shape[0]), :] = new_neighbor
        self.neighbor_size = self.neighbor_size + new_neighbor.shape[0]

        for point in points:
            if inside(point, image_shape) and not region_neighbor[tuple(point)] and not cur_region[tuple(point)]:
                region_neighbor[tuple(point)] = True


def compute_inner_boundary(self):
    """
        Compute the inner boundary
        """
        #Do something here.
        pass


def compute_external_boundary(self):
    """
        Compute the external boundary
        """
        #Do something here.
        pass


class RegionOptimize(Region):
    """
    Region optimizer.
    """

    def __init__(self, target_image, seeds, upperlimit, connectivity):
        """
        Parameters
        -----------------------------------------------------
        region_sequence: region sequence
        opt_measurement: the optimize measurement.
        mask_image: the mask image may be used in the compute process. which should be a ndarray type.
        prior_image:the prior image may be used in the compute process. which should be a ndarray type.
        """


class Aggregator:
    """
    Seeded region growing based on random seeds.
    """

    def __init__(self, seeds, region_sequence, raw_image, aggregator_type='average'):
        """
        Parameters
        -----------------------------------------------------
        region_sequence: A series of regions.
        raw_image: raw image.
        aggregator_type: 'average', 'magnitude', 'homogeneity', default is 'average'.
        """
        if not isinstance(region_sequence, np.ndarray):
            raise ValueError("The input region sequence  must be ndarray type. ")

        if not isinstance(raw_image, np.ndarray):
            raise ValueError("The input raw_image  must be ndarray type. ")

        if not isinstance(aggregator_type, str):
            raise ValueError("The value of aggregator_type must be str type. ")

    def set_seeds(self, seeds):
        self.seeds = seeds

    def get_seeds(self):
        return self.seeds

    def aggregator(self):
        """
        Aggregation for different regions
        """


class RandomSRG:
    """
    Seeded region growing based on random seeds.
    """

    def __init__(self, n_seeds, stop_criteria, ):
        """
        Parameters
        -----------------------------------------------------
        n_seeds: n seeds.
        stop_criteria: stop criteria about the n regions from n seeds.
        """
        if not isinstance(n_seeds, list):
            raise ValueError("The input seeds  must be list type. ")
        else:
            self.set_seeds(n_seeds)
        self.aggregator()


    def set_seeds(self, seeds):
        self.seeds = seeds

    def get_seeds(self):
        return self.seeds

    def set_stop_criteria(self, region, stop_type, stop_criteria):
        """
        Set the stop criteria.
        """
        self.stop_criteria = StopCriteria(region, stop_type, stop_criteria)

    def get_stop_criteria(self):
        """
        Return the stop criteria.
        """
        return self.stop_criteria

    def aggregator(self):
        """
        Aggregation for different regions
        """
        n_regions = None
        raw_image = None
        self.aggregator = Aggregator(self.seeds, n_regions, raw_image)
        self.aggregator.aggregator()


class AdaptiveSRG(SeededRegionGrowing):
    """
    Adaptive seeded region growing.
    """

    def __init__(self, target_image, seeds, upperlimit, connectivity):
        if not isinstance(seeds, np.ndarray):
            seeds = np.array(seeds)
        self.target_image = target_image
        self.set_seeds(seeds)
        self.connectivity = connectivity
        self.get_uplimit = upperlimit

    def growing(self):
        """
        Adaptive region growing.
        """
        image = self.target_image
        seeds = Seeds(self.seeds)
        similarity_criteria = NeighborSimilarity(metric='euclidean')
        connectivity = Connectivity('26')

        region_sequence = []
        child_vector = self.stop_criteria.computing
        for i in child_vector:
            stop_criteria_temp = StopCriteria(name='region_size', threshold=i)
            self.set_stop_criteria(stop_criteria_temp)
            region = self.grow()
            region_sequence.append(region)
        return region_sequence


if __name__ == "__main__":
    seed_coors = (((1, 2, 3), (3, 2, 1)), ((4, 5, 6), (6, 5, 1)))
    seeds3d = Seeds(seed_coors)
    print seeds3d.generating()

    randseeds = RandomSeeds(seed_coors)
    print randseeds.generating()


    #similarity_criteria = NeighborSimilarity(metric='euclidean',)
    #stop_criteria = StopCriteria(name='region_size', threshold=300)
    #connectivity = Connectivity('6')

