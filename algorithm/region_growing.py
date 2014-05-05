import numpy as np
import nibabel as nib
from utils import compute_offsets
from utils import inside
import random


class Seeds:
    """
    Seeds.
    """
    def __init__(self, seeds):
        """
        Init seeds.
        Parameters
        -----------------------------------------------------
        seeds: a set of coordinates or a region mask
        """
        if not isinstance(seeds, np.ndarray):
            if  isinstance(seeds, nib.nifti1.Nifti1Image) or isinstance(seeds, list):
                self.generating(seeds)
            else:
                raise ValueError("The value must be  a 1D/2D/3D ndarray or Nifti1Image format file.!")
        else:
            self.generating(seeds)

    def generating(self):
        """
        Generating new seeds.
        """
        return self.generating()

class RandomSeeds(Seeds):
    """
    Seeds.
    """
    def __init__(self, seeds, random_number=0):
        """
        Init seeds.
        Parameters
        -----------------------------------------------------
        seeds_type: 'separation', 'union', 'random'
        value: a set of coordinates or a region mask.
        """
        Seeds.__init__(seeds)
        self.seeds = seeds
        if not isinstance(random_number, int):
            raise ValueError("The random_number must be int type.")
        else:
            self.random_number = random_number

    def generating(self):
        """
        Generating new seeds.
        """
        if self.random_number == 0:
            return self.seeds
        else:
            return random.sample(self.seeds, self.random_number)


class SimilarityCriteria:
    """
    Distance measure.
    """
    def __init__(self, region, raw_image, similarity_type='difference', name='euclidean'):
        """
        Parameters
        -----------------------------------------------------
        region:The region growing.
        raw_image: The raw image.
        similarity_type:'size', 'intensity', 'homogeneity ' or 'deference'. Default is 'difference'.
        metric: 'educlidean', 'mahalanobis', 'minkowski','seuclidean', 'cityblock',ect. Default is 'euclidean'.
        """
        if not isinstance(region, np.ndarray):
            raise ValueError("The input region  must be ndarray type. ")

        if not isinstance(raw_image, np.ndarray):
            raise ValueError("The input raw_image  must be ndarray type. ")

        if not isinstance(name, str):
            raise ValueError("The value of name must be str type. ")

    def set_type(self, similarity_type):
        """
        Set the type of the distances which to use.
        """
        self.similarity_type = similarity_type

    def get_type(self):
        """
        Get the type the distance used in the region growing.
        """
        return self.similarity_type

    def set_name(self, name):
        """
        Set the name of the distances which to use.
        """
        self.name = name

    def get_name(self):
        """
        Get the name the distance used in the region growing.
        """
        return self.name

    def generating(self, region, raw_image):
        """
        Set the similarity metric.
        """
        from scipy.spatial.distance import pdist

        self.metric = pdist(self, region.flatten(), raw_image.flatten())


class StopCriteria:
    """
    Stop criteria.
    """
    def __init__(self, region, raw_image=None, stop_type='difference', value=None, mask_image=None):
        """
        Parameters
        -----------------------------------------------------
        region:The region growing.
        raw_image: The raw image.
        stop_type: 'size' , 'intensity', 'homogeneity' or 'deference'. Default is 'difference'.
        value: fixed value, it should be none when mode is equal to 'adaptive'.
        """
        if not isinstance(region, np.ndarray):
            raise ValueError("The input region  must be ndarray type. ")

        if not isinstance(raw_image, np.ndarray):
            raise ValueError("The input raw_image  must be ndarray type. ")

        if not isinstance(stop_type, str):
            raise ValueError("The name must be str type. ")
        elif stop_type is not 'size' and stop_type is not 'homogeneity' and \
              stop_type is not 'intensity' and stop_type is not 'difference':
            raise ValueError("The name must be 'size' or 'homogeneity'.")
        else:
            self.set_type(stop_type)

        if not isinstance(value, float) and not isinstance(value, int):
            raise ValueError("The value must be float or int type.")
        else:
            self.set_value(value)

    def set_type(self, stop_type):
        """
        Set the name of the stop criteria.
        """
        self.stop_type = stop_type

    def get_type(self):
        """
        Get the name of the stop criteria.
        """
        return self.stop_type

    def set_value(self, value):
        """
        Set the value of the stop criteria.
        """
        self.value = value

    def get_value(self):
        """
        Get the value of the stop criteria.
        """
        return self.value

    def generating(self, region, raw_image):
        """
        Set the similarity metric.
        """
        return self.generating(region, raw_image)


class RegionOptimizer:
    """
    Region optimizer.
    """
    def __init__(self, region_sequence, raw_image, name, optimizing, optpara=None):
        """
        Parameters
        -----------------------------------------------------
        region_sequence: A series of regions.
        raw_image: Raw image.
        name:The name of the region optimizer method.
        optimizing:
        optpara: Default is None.Optional parameter
        """
        if not isinstance(region_sequence, np.ndarray):
            raise ValueError("The input region sequence  must be ndarray type. ")

        if not isinstance(raw_image, np.ndarray):
            raise ValueError("The input raw image  must be np.ndarray type. ")

    def set_name(self, name):
        """
        Set the name of the region optimizer.
        """
        self.name = name

    def get_name(self):
        """
        Get the name of the  region optimizer.
        """
        return self.name

    def set_optimizing(self, optimizing):
        """
        Set the optimizing of the  region optimizer.
        """
        self.optimizing = optimizing

    def get_optimizing(self):
        """
        Get the optimizing of the  region optimizer.
        """
        return self.optimizing

    def set_optpara(self, optpara):
        """
        Set the optional parameter of the  region optimizer.
        """
        self.optpara = optpara

    def get_optpara(self):
        """
        Get the optional parameter of the  region optimizer.
        """
        return self.optpara


class SeededRegionGrowing:
    """
    Seeded region growing with a fixed threshold.
    """
    def __init__(self, target_image, seeds, stop_type, value=None, connectivity='8', similarity_criteria='euclidean', mask_image=None):
        """
        Parameters
        -----------------------------------------------------
        target_image: input image, a 2D/3D Nifti1Image format file
        seeds: a set of coordinates or a region mask
        value the stop threshold.
        """
        if isinstance(target_image, nib.nifti1.Nifti1Image):
            target_image = target_image.get_data()
            if len(target_image.shape) > 3 or len(target_image.shape) < 2:
                raise ValueError("Target image must be a 2D/3D or Nifti1Image format file.")
        elif isinstance(target_image, np.ndarray):
            if len(target_image.shape) > 3 or len(target_image.shape) < 2:
                raise ValueError("Target image must be a 2D/3D data.")

        elif isinstance(mask_image, np.ndarray):
            if len(target_image.shape) > 3 or len(target_image.shape) < 2:
                raise ValueError("Mask image must be a 2D/3D data.")
        else:
            raise ValueError("Must be a nifti1.Nifti1Image data format..")

        self.target_image = target_image
        self.set_seeds(seeds)
        self.set_stop_criteria(stop_type, value)
        self.set_connectivity(connectivity)

    def set_seeds(self, seeds):
        self.seeds = seeds

    def get_seeds(self):
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


class AdaptiveSRG(SeededRegionGrowing):
    """
    Adaptive seeded region growing.
    """
    def __init__(self, target_image, seed, Thres, connectivity):
        if not isinstance(seed, np.ndarray):
            seed = np.array(seed)
        self.target_image = target_image
        self.set_seeds(seed)
        self.get_seeds()
        self.git_thres = Thres
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
        for i in range(20,1000,20):
            region_list[i/20-1] = SeededRegionGrowing.grow()
        return region_list


    def region_optimizer(self, region_list, opt_measurement):
        contrast = []
        if opt_measurement != 'average' and opt_measurement != 'peripheral':
            raise ValueError("The optimize measurement must be average or peripheral contrast.")
        elif opt_measurement == 'average':
            for i in range(20,1000,20):
                contrast[i/20-1] = self.average_contrast()[i]
            k = np.array(contrast).argmax()
            return region_list[k]
        else:
            for i in range(20,1000,20):
                contrast[i/20-1] = self.peripheral_contrast()[i]
            k = np.array(contrast).argmax()
            return region_list[k]


class Average_contrast:
    """
    Max average contrast region growing.
    """
    def __init__(self, target_image, seed, Thres):
        if not isinstance(seed,np.ndarray):
            seed = np.array(seed)
        self.target_image = target_image
        self.set_seed(seed)
        self.set_stop_criteria(target_image, seed, Thres)

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
        self.stop_criteria = StopCriteria('size', 'fixed', number)

    def get_stop_criteria(self):
        """
        Return the stop criteria.
        """
        return self.stop_criteria

    def _grow(self, image, seed, Num):
        """
        Average contrast growing.
        """
        self.set_stop_criteria(image, seed, Num)
        N = self.get_stop_criteria().value
        return self.grow(image, seed, N)


class Peripheral_contrast:
    """
    Max peripheral contrast region growing.
    """
    def __init__(self, target_image, seed, Thres):
        if not isinstance(seed,np.ndarray):
            seed = np.array(seed)
        self.target_image = target_image
        self.set_seed(seed)
        self.set_stop_criteria(target_image, seed, Thres)

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
            if self.is_neiflag(flag_image,i,1):
                if inner_b == [ ]:
                    inner_b = i
                else:
                    inner_b= np.vstack((inner_b,i))
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
            inner_list[inner_pos] = [x,y,z,image[x,y,z]]
            for i in range(26):
                set0,set1,set2 = compute_offsets(3,26)[i]
                xn,yn,zn = x+set0,y+set1,z+set2
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
        self.stop_criteria = StopCriteria('size','fixed',number)

    def get_stop_criteria(self):
        """
        Return the stop criteria.
        """
        return self.stop_criteria

    def _grow(self, image, seed, Num):
        """
        Peripheral contrast growing.
        """
        self.set_stop_criteria(image, seed, Num)
        N = self.get_stop_criteria().value
        return self.grow(image, seed, N)


if __name__ == "__main__":
    t_image = nib.load('../data/S2/tstat1.nii.gz')
    data = t_image.get_data()
    A = Average_contrast(data, (26,38,25), 1000)
    new_image = A._grow(data, (26,38,25), 1000)
    t_image._data = new_image
    nib.save(t_image,'ACB_S2_image.nii.gz')
    print 'average contrast growing has been saved.'

    #B = Peripheral_contrast(data, (26,38,25), 1000)
    #new_image = B._grow(data, (26,38,25), 1000)
    #t_image._data = new_image
    #nib.save(t_image,'PCB_S2_image')
    #print 'peripheral contrast growing has been saved.'

    #C = fixed_region_grow(data,(26,38,25),200)
    #new_image = C._grow(data,(26,38,25),200)
    #t_image._data = new_image
    #nib.save(t_image,'fixed_thres_S2_image.nii.gz')
    #print 'fixed threshold growing has been saved.'



























