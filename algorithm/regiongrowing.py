import numpy as np
import nibabel as nib
from connectivity import compute_offsets
from utils import inside
import random


class SeededRegionGrowing:
    """
    Seeded region growing with a fixed threshold.
    """
    # def __init__(self, target_image, seeds, stop_type, value=None, connectivity='8', similarity_criteria='euclidean', mask_image=None):
    def __init__(self, seeds, connectivity, similarity_criteria, stop_criteria, region_sequence):
        """
        Parameters
        -----------------------------------------------------
        target_image: input image, a 2D/3D Nifti1Image format file
        seeds: a set of coordinates or a region mask
        value the stop threshold.
        """
        if isinstance(region_sequence, nib.nifti1.Nifti1Image):
            target_image = region_sequence.get_data()
            if len(target_image.shape) > 4 or len(target_image.shape) < 2:
                raise ValueError("Target image must be a 2D/3D or Nifti1Image format file.")
        elif isinstance(region_sequence, np.ndarray):
            if len(region_sequence.shape) > 4 or len(region_sequence.shape) < 2:
                raise ValueError("Target image must be a 2D/3D data.")
        else:
            raise ValueError("Must be a nifti1.Nifti1Image data format..")

        self.target_image = target_image
        self.set_seeds(seeds)
        self.set_stop_criteria(stop_criteria)
        self.set_similarity_criteria(similarity_criteria)
        self.set_connectivity(connectivity)

    def set_seeds(self, seeds):
        self.seeds = seeds

    def get_seeds(self):
        return self.seeds

    def set_stop_criteria(self, region, stop_criteria):
        """
        Set the stop criteria.
        """
        self.stop_criteria = stop_criteria

    def get_stop_criteria(self):
        """
        Return the stop criteria.
        """
        return self.stop_criteria

    def set_connectivity(self, connectivity='6'):
        """
        Set the connectivity.
        """
        self.connectivity = compute_offsets(len(self.target_image.shape), int(connectivity))

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

    def grow(self):
        """
        Fixed threshold region growing.
        """
        seeds = self.get_seeds()[0]
        image_shape = self.target_image.shape

        if not inside(np.array(seeds), image_shape):
            raise ValueError("The seed is out of the image range.")

        region_size = 1
        origin_t = self.target_image[tuple(seeds)]
        tmp_image = np.zeros_like(self.target_image)
        self.inner_image = np.zeros_like(self.target_image)

        neighbor_free = 10000
        neighbor_pos = -1
        neighbor_list = np.zeros((neighbor_free, len(image_shape) + 1))

        while region_size <= self.stop_criteria.get_value():
            for i in range(0, self.get_connectivity().shape[1]):
                seedn = (np.array(seeds) + self.get_connectivity()[i]).tolist()
                if inside(seedn, image_shape) and tmp_image[tuple(seedn)] == 0:
                    neighbor_pos = neighbor_pos + 1
                    neighbor_list[neighbor_pos][0:len(image_shape)] = seedn
                    neighbor_list[neighbor_pos][len(image_shape)-1] = self.target_image[tuple(seedn)]
                    tmp_image[tuple(seedn)] = 1

            tmp_image[tuple(seeds)] = 2
            self.inner_image[tuple(seeds)] = self.target_image[tuple(seeds)]
            region_size += 1

            distance = np.abs(neighbor_list[:neighbor_pos + 1, len(image_shape)] - np.tile(origin_t, neighbor_pos + 1))
            index = distance.argmin()
            seed = neighbor_list[index][:len(image_shape)]
            neighbor_list[index] = neighbor_list[neighbor_pos]
            neighbor_pos -= 1
        return self.inner_image


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
        metric: 'educlidean', 'mahalanobis', 'minkowski','seuclidean', 'cityblock',ect. Default is 'euclidean'.
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
    def __init__(self, metric='size'):
        """
        Parameters
        -----------------------------------------------------
        metric: 'educlidean', 'mahalanobis', 'minkowski','seuclidean', 'cityblock',ect. Default is 'euclidean'.
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

    def computing(self, region, raw_image=None, mask_image=None, prior_image=None):
        """
        Compute the neighbor similarity.
        Parameters
        -----------------------------------------------------
        region:The region growing.
        raw_image: The raw image.
        mask_image: the mask image may be used in the compute process. which should be a ndarray type.
        prior_image:the prior image may be used in the compute process. which should be a ndarray type.
        """
        if self.metric is 'educlidean':
            pass
        elif self.metric is 'mahalanobis':
            pass
        elif self.metric is 'minkowski':
            pass
        elif self.metric is 'seuclidean':
            pass
        elif self.metric is 'cityblock':
            pass
        else:
            return None

class StopCriteria(object):
    """
    Stop criteria.
    """
    def __init__(self, name='region_homogeneity', threshold=None):
        """
        Parameters
        -----------------------------------------------------
        name:'region_homogeneity','region_morphology','region_difference', default is 'region_difference'
        threshold: a int value or None, default is None which means the adaptive method will be used.
        """
        if not isinstance(name, str):
            raise ValueError("The name of the stop criteria should be str type.")

        if not isinstance(threshold, int) and threshold is None:
            raise ValueError("The threshold of the stop criteria should be int type or None.")

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
        Set the similarity metric.
        """
        #Do something here.
        pass


class Region(object):
    """
    Distance measure.
    """
    def __init__(self, seed, cur_region):
        """
        Parameters
        -----------------------------------------------------
        seed: the seed to generate a region, which should be ndarray type.
        cur_region: the current region.
        """
        if not isinstance(seed, np.ndarray):
            raise ValueError("The seed of the Region class  must be ndarray type. ")
        else:
            self.seed = seed

        if not isinstance(cur_region, np.ndarray):
            raise ValueError("The current region of the Region class must be ndarray type. ")
        else:
            self.cur_region = cur_region

    def get_neighbor(self):
        """
        Get the neighbor.
        """
        return self.neighbor

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


class RegionOptimizer:
    """
    Region optimizer.
    """
    def __init__(self, raw_image, name, mask_image=None, prior_image=None):
        """
        Parameters
        -----------------------------------------------------
        raw_image: raw image.
        name: the name of the optimizer.
        mask_image: the mask image may be used in the compute process. which should be a ndarray type.
        prior_image:the prior image may be used in the compute process. which should be a ndarray type.
        """
        if not isinstance(raw_image, np.ndarray):
            raise ValueError("The input region sequence  must be ndarray type. ")

        self.raw_image = raw_image
        self.name = name
        self.mask_image = mask_image
        self.prior_image = prior_image

    def optimize(self):
        """
        Get the optimize region.
        """
        return self.optimize()


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
    def __init__(self, target_image, seed, upperlimit, connectivity):
        if not isinstance(seed, np.ndarray):
            seed = np.array(seed)
        self.target_image = target_image
        self.set_seeds(seed)
        self.get_seeds()
        self.get_uplimit = upperlimit
        self.set_connectivity(connectivity)
        self.get_connectivity()

    def set_seeds(self, seeds):
        """
        Set the seeds.
        """
        self.seeds = seeds

    def get_seeds(self):
        """
        Get the seeds.
        """
        return self.seeds

    def set_stop_criteria(self, stop_type, stop_criteria):
        """
        Set the stop criteria.
        """
        self.stop_criteria = StopCriteria(stop_type, stop_criteria)

    def get_stop_criteria(self):
        """
        Return the stop criteria.
        """
        return self.stop_criteria

    def set_connectivity(self, connectivity):
        """
        Set the connectivity.
        """
        self.connectivity = compute_offsets(len(self.target_image.shape), int(connectivity))

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

    def average_contrast(self):
        """
        return average contrast list.
        """
        return self.average_contrast()

    def peripheral_contrast(self):
        """
        return peripheral contrast list.
        """
        return self.peripheral_contrast()

    def grow(self):
        """
        Adaptive region growing.
        """
        region_list = []
        for i in range(20,self.get_uplimit,20):
            region_list[i/20-1] = SeededRegionGrowing.grow()
        return region_list


    def region_optimizer(self, region_list, opt_measurement):
        contrast = []
        if opt_measurement != 'average' and opt_measurement != 'peripheral':
            raise ValueError("The optimize measurement must be average or peripheral contrast.")
        elif opt_measurement == 'average':
            for i in range(20,self.get_uplimit,20):
                contrast[i/20-1] = self.average_contrast()[i]
            k = np.array(contrast).argmax()
            return region_list[k]
        else:
            for i in range(20,self.get_uplimit,20):
                contrast[i/20-1] = self.peripheral_contrast()[i]
            k = np.array(contrast).argmax()
            return region_list[k]


class AverageContrast:
    """
    Max average contrast region growing.
    """
    def __init__(self, target_image, seeds, thres):
        if not isinstance(seeds, np.ndarray):
            seeds = np.array(seeds)
        self.thres = thres
        self.target_image = target_image
        self.set_seeds(seeds)

    def set_seeds(self, seeds):
        """
        Set the seeds.
        """
        self.seeds = seeds

    def get_seeds(self):
        """
        Get the seeds.
        """
        return self.seeds

    def set_stop_criteria(self, image, seeds, Num):
        """
        set stop criteria according to the max average contrast point.
        """
        x,y,z = seeds
        image_shape = image.shape
        if inside(seeds,image_shape)!=True:
            print "The seed is out of the image range."
            return False

        contrast = []
        region_size = 1
        origin_t = image[x,y,z]
        inner_list = [origin_t]
        tmp_image = np.zeros_like(image)

        neighbor_free = 10000
        neighbor_pos = -1
        neighbor_list = np.zeros((neighbor_free,4))

        while region_size <= Num:
            for i in range(26):
                set0,set1,set2 = compute_offsets(3,26)[i]
                xn,yn,zn = x+set0,y+set1,z+set2
                if inside((xn,yn,zn),image_shape) and tmp_image[xn,yn,zn]==0:
                    neighbor_pos = neighbor_pos + 1
                    neighbor_list[neighbor_pos] = [xn,yn,zn,image[xn,yn,zn]]
                    tmp_image[xn,yn,zn] = 1

            out_boundary = neighbor_list[np.nonzero(neighbor_list[:,3]),3]
            contrast = contrast + [np.mean(np.array(inner_list)) - np.mean(out_boundary)]

            tmp_image[x,y,z] = 2
            region_size += 1

            if (neighbor_pos+100 > neighbor_free):
                neighbor_free +=10000
                new_list = np.zeros((10000,4))
                neighbor_list = np.vstack((neighbor_list,new_list))
                #if the longth of neighbor_list is not enough,add another 10000

            distance = np.abs(neighbor_list[:neighbor_pos+1,3] - np.tile(origin_t,neighbor_pos+1))
            index = distance.argmin()
            x,y,z = neighbor_list[index][:3]
            inner_list = inner_list + [image[x,y,z]]
            neighbor_list[index] = neighbor_list[neighbor_pos]
            neighbor_pos -= 1
        number = int(np.array(contrast).argmax()+1)
        print number
        self.stop_criteria = StopCriteria('region_homogeneity', number)

    def get_stop_criteria(self):
        """
        Return the stop criteria.
        """
        return self.stop_criteria

    def grow(self):
        """
        Give a coordinate ,return a region.
        """
        seeds = self.get_seeds()
        image = self.target_image
        x,y,z = seeds
        image_shape = image.shape

        if inside(seeds,image_shape)!=True:
            print "The seed is out of the image range."
            return False

        region_size = 1
        origin_t = image[x,y,z]

        Num = self.set_stop_criteria(image, seeds, self.thres)
        tmp_image = np.zeros_like(image)
        inner_image = np.zeros_like(image)

        neighbor_free = 10000
        neighbor_pos = -1
        neighbor_list = np.zeros((neighbor_free,4))

        while region_size <= Num:
            for i in range(26):
                set0,set1,set2 = compute_offsets(3,26)[i]
                xn,yn,zn = x+set0,y+set1,z+set2
                if inside((xn,yn,zn),image_shape) and tmp_image[xn,yn,zn]==0:
                    neighbor_pos = neighbor_pos+1
                    neighbor_list[neighbor_pos] = [xn,yn,zn,image[xn,yn,zn]]
                    tmp_image[xn,yn,zn] = 1

            tmp_image[x,y,z] = 2
            inner_image[x,y,z] = image[x,y,z]
            region_size += 1

            distance = np.abs(neighbor_list[:neighbor_pos+1,3] - np.tile(origin_t,neighbor_pos+1))
            index = distance.argmin()
            x,y,z = neighbor_list[index][:3]
            neighbor_list[index] = neighbor_list[neighbor_pos]
            neighbor_pos -= 1

        return inner_image


class PeripheralContrast:
    """
    Max peripheral contrast region growing.
    """
    def __init__(self, target_image, seeds, thres):
        if not isinstance(seeds,np.ndarray):
            seeds = np.array(seeds)
        self.thres = thres
        self.target_image = target_image
        self.set_seeds(seeds)
        self.set_stop_criteria(target_image, seeds, thres)

    def is_neiflag(self,flag_image,coordinate,flag):
        """
        if coordinate has a neighbor with certain flag return True,else False.
        """
        x,y,z = coordinate
        for j in range(26):
            set0,set1,set2 = compute_offsets(3,26)[j]
            xn,yn,zn = x+set0,y+set1,z+set2
            if flag_image[xn,yn,zn]==flag:
                return True
        return False

    def inner_boundary(self,flag_image,inner_region_cor):
        """
        find the inner boundary of the region.
        """
        inner_b = []
        for i in inner_region_cor:
            if self.is_neiflag(flag_image, i, 1):
                if inner_b == []:
                    inner_b = i
                else:
                    inner_b= np.vstack((inner_b, i))
        return np.array(inner_b)

    def set_stop_criteria(self, image, seed, Num):
        """
        set stop criteria according to the max average contrast point.
        """
        x,y,z = seed
        image_shape = image.shape
        if inside(seed,image_shape)!=True:
            print "The seed is out of the image range."
            return False

        contrast = []
        region_size = 1
        origin_t = image[x,y,z]
        tmp_image = np.zeros_like(image)

        default_space = 10000
        outer_pos = -1
        inner_pos = -1
        inner_list = np.zeros((default_space,4))
        outer_boundary_list = np.zeros((default_space,4))

        while region_size <= Num:
            inner_pos = inner_pos + 1
            inner_list[inner_pos] = [x, y, z,image[x, y, z]]
            for i in range(26):
                set0,set1,set2 = compute_offsets(3, 26)[i]
                xn,yn,zn = x + set0,y + set1,z + set2
                if inside((xn,yn,zn),image_shape) and tmp_image[xn,yn,zn]==0:
                    outer_pos = outer_pos+1
                    outer_boundary_list[outer_pos] = [xn,yn,zn,image[xn,yn,zn]]
                    tmp_image[xn,yn,zn] = 1

            outer_boundary = outer_boundary_list[np.nonzero(outer_boundary_list[:,3]),3]
            inner_region_cor = inner_list[np.nonzero(inner_list[:,3]),:3][0]
            inner_boundary_cor = self.inner_boundary(tmp_image,np.array(inner_region_cor))

            inner_boundary_val = []
            if len(inner_boundary_cor.shape) == 1:
                inner_boundary_val = inner_boundary_val + [image[inner_boundary_cor[0], \
                                     inner_boundary_cor[1],inner_boundary_cor[2]]]
            else:
                for i in inner_boundary_cor:
                    inner_boundary_val = inner_boundary_val + [image[i[0],i[1],i[2]]]

            contrast = contrast + [np.mean(inner_boundary_val) - np.mean(outer_boundary)]
            tmp_image[x,y,z] = 2
            region_size += 1

            if (outer_pos+100 > default_space):
                default_space +=10000
                new_list = np.zeros((10000,4))
                outer_boundary_list = np.vstack((outer_boundary_list,new_list))

            distance = np.abs(outer_boundary_list[:outer_pos+1,3] - np.tile(origin_t,outer_pos+1))
            index = distance.argmin()
            x,y,z = outer_boundary_list[index][:3]

            outer_boundary_list[index] = outer_boundary_list[outer_pos]
            outer_pos -= 1

        number = int(np.array(contrast).argmax()+1)
        print number
        self.stop_criteria = StopCriteria('region_homogeneity',number)

    def get_stop_criteria(self):
        """
        Return the stop criteria.
        """
        return self.stop_criteria

    def grow(self):
        """
        Give a coordinate ,return a region.
        """
        seeds = self.get_seeds()
        image = self.target_image
        x,y,z = seeds
        image_shape = image.shape

        if inside(seeds,image_shape)!=True:
            print "The seed is out of the image range."
            return False

        region_size = 1
        origin_t = image[x,y,z]

        Num = self.set_stop_criteria(image, seeds, self.thres)
        tmp_image = np.zeros_like(image)
        inner_image = np.zeros_like(image)

        neighbor_free = 10000
        neighbor_pos = -1
        neighbor_list = np.zeros((neighbor_free,4))

        while region_size <= Num:
            for i in range(26):
                set0,set1,set2 = compute_offsets(3,26)[i]
                xn,yn,zn = x+set0,y+set1,z+set2
                if inside((xn,yn,zn),image_shape) and tmp_image[xn,yn,zn]==0:
                    neighbor_pos = neighbor_pos+1
                    neighbor_list[neighbor_pos] = [xn,yn,zn,image[xn,yn,zn]]
                    tmp_image[xn,yn,zn] = 1

            tmp_image[x,y,z] = 2
            inner_image[x,y,z] = image[x,y,z]
            region_size += 1

            distance = np.abs(neighbor_list[:neighbor_pos+1,3] - np.tile(origin_t,neighbor_pos+1))
            index = distance.argmin()
            x,y,z = neighbor_list[index][:3]
            neighbor_list[index] = neighbor_list[neighbor_pos]
            neighbor_pos -= 1

        return inner_image



























