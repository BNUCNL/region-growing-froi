import numpy as np
import nibabel as nib
from connectivity import compute_offsets
from utils import inside
import random


class SeededRegionGrowing:
    """
    Seeded region growing with a fixed threshold.
    """
    def __init__(self, target_image, seeds, similarity_criteria, stop_criteria, connectivity, region_sequence=None):
        """
        Parameters
        -----------------------------------------------------
        target_image: input image, a 2D/3D Nifti1Image format file
        seeds: a set of coordinates or a region mask
        value the stop threshold.
        """
        if isinstance(target_image, nib.nifti1.Nifti1Image):
            self.target_image = target_image.get_data()
            if len(target_image.shape) > 4 or len(target_image.shape) < 2:
                raise ValueError("Target image must be a 2D/3D or Nifti1Image format file.")
        elif isinstance(target_image, np.ndarray):
            self.target_image = target_image
            if len(target_image.shape) > 4 or len(target_image.shape) < 2:
                raise ValueError("Target image must be a 2D/3D or Nifti1Image format file.")

        self.seeds = seeds
        self.similarity_criteria = similarity_criteria
        self.stop_criteria = stop_criteria
        self.connectivity = connectivity
        self.region_sequence = region_sequence

    def set_target_image(self, target_image):
        self.target_image = target_image

    def get_target_image(self):
        return self.target_image

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

    def set_connectivity(self, connectivity):
        """
        Set the connectivity.
        """
        self.connectivity = connectivity

    def get_connectivity(self):
        """
        Get the connectivity.
        """
        return self.connectivity

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

    def set_region_sequence(self, region_sequence):
        """
        Set the region sequence.
        """
        self.region_sequence = region_sequence

    def get_region_sequence(self):
        """
        Get the region sequence..
        """
        return self.region_sequence

    def grow(self):
        """
        Fixed threshold region growing.
        """
        if len(self.seeds.coords.shape) == 1:
            self.seeds.coords = np.array([self.seeds.coords])
        start_point = self.seeds.coords[0]

        image_shape = self.target_image.shape
        if len(image_shape) == 4:
            image_shape = self.target_image.shape[:3]
            self.clone_image = self.target_image[..., 0].copy()
        else:
            self.clone_image = self.target_image

        region_neighbor = np.zeros_like(self.clone_image, dtype=np.bool)
        cur_region = np.zeros_like(self.clone_image, dtype=np.bool)
        region_size = 1
        region = Region(region_neighbor, cur_region)

        for i in range(self.seeds.coords.shape[0]):
            cur_region[tuple(self.seeds.coords[i])] = True
            if not inside(np.array(tuple(start_point)), image_shape):
                raise ValueError("The seed is out of the image range.")

        threshold = self.stop_criteria.computing()
        while region_size <= threshold:
            points = self.connectivity.computing(start_point, image_shape)
            for point in points:
                if inside(point, image_shape) and not region_neighbor[tuple(point)] and not cur_region[tuple(point)]:
                    region_neighbor[tuple(point)] = True
            cur_region[tuple(start_point)] = True
            start_point = self.similarity_criteria.computing(region, self.target_image)
            region_size += 1

        return region


class Seeds:
    """
    Seeds.
    """
    def __init__(self, coords):
        """
        Init seeds.
        Parameters
        -----------------------------------------------------
        seeds: a set of coordinates or a region mask
        """
        if not isinstance(coords, np.ndarray):
            raise ValueError("The value must be  a 1D/2D/3D!")
        else:
            self.coords = coords

    def generating(self):
        """
        Generating new seeds.
        """
        return self.generating()


class RandomSeeds(Seeds):
    """
    Random Seeds.
    """
    def __init__(self, seeds, random_number=0):
        """
        Init seeds.
        Parameters
        -----------------------------------------------------
        seeds_type: 'separation', 'union', 'random'
        value: a set of coordinates or a region mask.
        """
        Seeds.__init__(self, seeds)

        if not isinstance(random_number, int):
            raise ValueError("The random_number must be int type.")
        else:
            self.random_number = random_number

    def generating(self):
        """
        Generating new seeds.
        """
        if self.random_number == 0:
            return self.seeds.coords
        else:
            return random.sample(self.seeds.coords, self.random_number)


class SimilarityCriteria:
    """
    Similarity criteria..
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

    def computing(self, region, raw_image=None, mask_image=None, prior_image=None):
        """
        Compute the  similarity.
        Parameters
        -----------------------------------------------------
        region:The region growing.
        raw_image: The raw image.
        mask_image: the mask image may be used in the compute process. which should be a ndarray type.
        prior_image:the prior image may be used in the compute process. which should be a ndarray type.
        """
        return self.computing()


class HomogeneitySimilarity(SimilarityCriteria):
    """
    Homogeneity similarity.
    """
    def __init__(self, metric='standard_deviation'):
        """
        Parameters
        -----------------------------------------------------
        metric: 'standard_deviation', 'kendell_cc', 'mean_cross_correlation', default is 'standard_deviation'.
        """
        SimilarityCriteria.__init__(self, metric)

    def set_metric(self, metric):
        """
        Set the metric of the homogeneity similarity...
        """
        self.metric = metric

    def get_metric(self):
        """
        Get the metric of the homogeneity similarity...
        """
        return self.metric

    def computing(self, region, raw_image=None, mask_image=None, prior_image=None):
        """
        Compute the homogeneity similarity.
        Parameters
        -----------------------------------------------------
        region:The region growing.
        raw_image: The raw image.
        mask_image: the mask image may be used in the compute process. which should be a ndarray type.
        prior_image:the prior image may be used in the compute process. which should be a ndarray type.
        """
        if self.metric is 'standard_deviation':
            pass
        elif self.metric is 'kendell_cc':
            pass
        elif self.metric is 'mean_cross_correlation':
            pass
        else:
            return None


class MorphologySimilarity(SimilarityCriteria):
    """
    Morphology similarity.
    """
    def __init__(self, metric='size'):
        """
        Parameters
        -----------------------------------------------------
        metric: 'size', 'volume', 'shape', default is 'size'
        """
        SimilarityCriteria.__init__(self, metric)

    def set_metric(self, metric):
        """
        Set the metric of the morphology similarity...
        """
        self.metric = metric

    def get_metric(self):
        """
        Get the metric of the morphology similarity...
        """
        return self.metric

    def computing(self, region, raw_image=None, mask_image=None, prior_image=None):
        """
        Compute the morphology similarity.
        Parameters
        -----------------------------------------------------
        region:The region growing.
        raw_image: The raw image.
        mask_image: the mask image may be used in the compute process. which should be a ndarray type.
        prior_image:the prior image may be used in the compute process. which should be a ndarray type.
        """
        if self.metric is 'size':
            pass
        elif self.metric is 'volume':
            pass
        elif self.metric is 'shape':
            pass
        else:
            return None


class NeighborSimilarity(SimilarityCriteria):
    """
    Neighbor similarity.
    """
    def __init__(self, metric='educlidean'):
        """
        Parameters
        -----------------------------------------------------
        metric: 'euclidean', 'mahalanobis', 'minkowski','seuclidean', 'cityblock',ect. Default is 'euclidean'.
        """
        SimilarityCriteria.__init__(self, metric)

    def set_metric(self, metric):
        """
        Set the metric of the neighbor similarity...
        """
        self.metric = metric

    def get_metric(self):
        """
        Get the metric of the neighbor similarity...
        """
        return self.metric

    def computing(self, region, raw_image, mask_image=None, prior_image=None):
        """
        Compute the neighbor similarity.
        Parameters
        -----------------------------------------------------
        region:The region growing.
        raw_image: The raw image.
        mask_image: the mask image may be used in the compute process. which should be a ndarray type.
        prior_image:the prior image may be used in the compute process. which should be a ndarray type.
        """
        from scipy.spatial import distance

        cur_region = region.get_current_region()
        region_neighbor = region.get_region_neighbor()
        region_mean = np.mean(raw_image[cur_region], axis=0)
        temp_region = np.zeros_like(cur_region) + 10000
        temp_region[region_neighbor] = distance.cdist(region_mean.reshape(1, region_mean.size), raw_image
                                  [region_neighbor].reshape(region_neighbor.sum(), region_mean.size), self.metric)[0, :]
        index = np.unravel_index(temp_region.argmin(), cur_region.shape)

        return index


class StopCriteria(object):
    """
    Stop criteria.
    """
    def __init__(self, name='region_size', threshold=None):
        """
        Parameters
        -----------------------------------------------------
        name:'region_homogeneity','region_morphology','region_difference', 'region_size', default is 'region_difference'
        threshold: a int value or None, default is None which means the adaptive method will be used.
        """
        if not isinstance(name, str):
            raise ValueError("The name of the stop criteria should be str type.")

        if not isinstance(threshold, int) and threshold is None:
            raise ValueError("The threshold of the stop criteria should be int type or None.")

        self.name = name
        self.threshold = threshold

    def set_name(self, name):
        """
        Get the name of the stop criteria..
        """
        self.metric = name

    def get_name(self):
        """
        Get the name of the stop criteria..
        """
        return self.name


    def computing(self):
        """
        Set the stop criteria.
        """
        return self.threshold


class Connectivity:
    """
    Pixel connectivity.
    """
    def __init__(self, name='8'):
        """
        Parameters
        -----------------------------------------------------
        name:for 2D ndarray, the name can be '4', '8'; for 3d ndarray, the name can be '6', '18', '26', default is '8'.
        """
        self.name = name

    def set_name(self, name):
        """
        Get the name of the connectivity..
        """
        self.name = name

    def get_name(self):
        """
        Get the name of the connectivity..
        """
        return self.name

    def computing(self, point, image_shape):
        """
        Get the connectivity of the connectivity..
        Parameters
        -----------------------------------------------------
        point: The point of the image, pixel or voxel.
        image_shape: The shape of  input target image.
        """
        return point + compute_offsets(len(image_shape), int(self.name))


class Region(object):
    """
    Distance measure.
    """
    def __init__(self, region_neighbor, cur_region):
        """
        Parameters
        -----------------------------------------------------
        cur_region: the current region.
        """
        if not isinstance(region_neighbor, np.ndarray):
            raise ValueError("The neighbor of the Region class must be ndarray type. ")
        if not isinstance(cur_region, np.ndarray):
            raise ValueError("The current region of the Region class must be ndarray type. ")

        self.region_neighbor = region_neighbor
        self.cur_region = cur_region

    def set_current_region(self, cur_region):
        """
        Get the neighbor.
        """
        self.cur_region = cur_region

    def get_current_region(self):
        """
        Get the neighbor.
        """
        return self.cur_region

    def set_region_neighbor(self, region_neighbor):
        """
        Set the region neighbor.
        """
        self.region_neighbor = region_neighbor

    def get_region_neighbor(self):
        """
        Get the  region neighbor.
        """
        return self.region_neighbor

    def compute_IB(self):
        """
        Compute the inner boundary
        """
        #Do something here.
        pass

    def compute_EB(self):
        """
        Compute the external boundary
        """
        #Do something here.
        pass


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
    def __init__(self, n_seeds, stop_criteria,):
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
        child_vector = self.stop_criteria.computing()
        for i in child_vector:
            stop_criteria_temp = StopCriteria(name='region_size', threshold=i)
            self.set_stop_criteria(stop_criteria_temp)
            region = self.grow()
            region_sequence.append(region)
        return region_sequence

class RegionOptimize(Region):
    """
    Region optimizer.
    """



