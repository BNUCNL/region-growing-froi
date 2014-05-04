import numpy as np
import nibabel as nib
from utils import compute_offsets
from utils import inside


class RegionGrowing:
    """
    Base class in region growing.

    """
    def __init__(self, target_image, seed, stop_criteria, connectivity='8', similarity_criteria='euclidean', mask_image=None):
        """
         Parameters
        ----------
        input: must be 2D/3D/4D np.ndarray type or a Nifti1 format file(*.nii, *.nii.gz).
        seed: the seed points.
        stop_criteria: The stop criteria of region growing to stop.
        """
        if not isinstance(target_image, nib.nifti1.Nifti1Image):
            if len(target_image.get_shape()) > 4 or len(target_image.get_shape()) < 2:
                raise ValueError("Must be Nifti1Image format file.")
        elif len(target_image.shape) > 4 or len(target_image.shape) < 2:
            raise ValueError("Must be a 2D/3D/4D data.")

        if not isinstance(seed, np.ndarray):
            self.seed = np.array(seed)

    def set_seed(self, seed):
        """
        Set the seed points.
        """
        self.seed = seed

    def get_seed(self):
        """
        Return the seed points.
        """
        return self.seed

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

    def set_connectivity(self, connectivity):
        """
        Set the connectivity.
        """
        return self.set_connectivity(connectivity)

    def get_connectivity(self):
        """
        Get the connectivity.
        """
        return self.get_connectivity()

    def set_stop_criteria(self, stop_criteria):
        """
        Set the stop criteria.
        """
        return self.set_stop_criteria(stop_criteria)

    def get_stop_criteria(self):
        """
        Return the stop criteria.
        """
        return self.get_stop_criteria()

    def grow(self):
        """
        Give a coordinate ,return a region.
        """
        return self.grow()


class Seeds:
    """
    Seeds.
    """
    def __init__(self, seeds_type, value):
        """
        Init seeds.
        Parameters
        -----------------------------------------------------
        seeds_type: 'separation', 'union', 'random'
        value: a set of coordinates or a region mask.
        """
        if seeds_type is not 'separation' or seeds_type is not 'union' or seeds_type is not 'random':
            raise ValueError("The input seeds type error!")
        else:
            self.set_seeds_type(seeds_type)

        if not isinstance(value, np.ndarray):
            if  isinstance(value, nib.nifti1.Nifti1Image):
                self.set_seeds_value(value)
            else:
                raise ValueError("The value must be  a 1D/2D/3D ndarray or Nifti1Image format file.!")

        else:
            self.set_seeds_value(value)

    def set_seeds_type(self, seeds_type):
        """
        Set the seeds type.
        """
        self.seeds_type = seeds_type

    def get_seeds_type(self):
        """
        Get the seeds type.
        """
        return self.seeds_type

    def set_seeds_value(self, value):
        """
        Set the seeds type.
        """
        self.value = value

    def get_seeds_value(self):
        """
        Get the seeds value.
        """
        return self.value


class SimilarityCriteria:
    """
    Distance measure.
    """
    def __init__(self, region, raw_image, similarity_type='difference', name='euclidean', reference='seed'):
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
        self.set_metric(region, raw_image)

        if not isinstance(name, str):
            raise ValueError("The value of name must be str type. ")
        else:
            self.set_metric(name)

        if not isinstance(reference, str):
            raise ValueError("The value of reference must be str type. ")
        else:
            self.set_reference(reference)

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

    def set_metric(self, region, raw_image):
        """
        Set the similarity metric.
        """
        from scipy.spatial.distance import pdist

        self.metric = pdist(self, region.flatten(), raw_image.flatten())

    def get_metric(self):
        """
        Get the similarity metric.
        """
        return self.metric

    def set_reference(self, reference):
        """
        Set the similarity reference.
        """
        self.reference = reference

    def get_reference(self):
        """
        Get the similarity reference.
        """
        return self.reference


class StopCriteria:
    """
    Stop criteria.
    """
    def __init__(self, stop_type='difference', value=None):
        """
        Parameters
        -----------------------------------------------------
        stop_type: 'size' , 'intensity', 'homogeneity' or 'deference'. Default is 'difference'.
        value: fixed value, it should be none when mode is equal to 'adaptive'.
        """
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


class FixedThresholdSRG(RegionGrowing):
    """
    Region growing with a fixed threshold.
    """
    def __init__(self, target_image, seed, value, connectivity='6'):
        """
        Parameters
        -----------------------------------------------------
        target_image: input image, a 2D/3D Nifti1Image format file
        seed: the seed points.
        value the stop threshold.
        """
        RegionGrowing.__init__(self)
        if not isinstance(target_image, nib.nifti1.Nifti1Image):
            if len(target_image.get_shape()) > 3 or len(target_image.get_shape()) < 2:
                raise ValueError("Must be a 2D/3D or Nifti1Image format file.")
            else:
                target_image = target_image.get_data()
        elif len(target_image.shape) > 3 or len(target_image.shape) < 2:
            raise ValueError("Must be a 2D/3D data.")

        self.target_image = target_image
        self.set_seed(seed)
        self.set_stop_criteria(value)
        self.set_connectivity(connectivity)

    def set_seed(self, seed):
        self.seed = seed

    def get_seed(self):
        return self.seed


    def set_stop_criteria(self, stop_criteria):
        """
        Set the stop criteria.
        """
        self.stop_criteria = StopCriteria('size', 'fixed', stop_criteria)

    def get_stop_criteria(self):
        """
        Return the stop criteria.
        """
        return self.stop_criteria

    def set_connectivity(self, connectivity='6'):
        """
        Set the connectivity.
        """
        self.connectivity = compute_offsets(len(self.target_image), int(connectivity))

    def get_connectivity(self):
        """
        Get the connectivity.
        """
        return self.connectivity

    def grow(self):
        """
        Fixed threshold region growing.
        """
        seed = self.get_seed()
        image_shape = self.target_image

        if inside(self.get_seed(), image_shape):
            raise ValueError("The seed is out of the image range.")

        region_size = 1
        origin_t = self.target_image[seed]
        tmp_image = np.zeros_like(self.target_image)
        self.inner_image = np.zeros_like(self.target_image)

        neighbor_free = 10000
        neighbor_pos = -1
        neighbor_list = np.zeros((neighbor_free, 4))

        while region_size <= self.stop_criteria.get_value():
            for i in range(self.get_connectivity()):
                seedn = (np.array(seed) + self.get_connectivity()[i]).tolist()
                if inside(seedn, image_shape) and tmp_image[seedn] == 0:
                    neighbor_pos = neighbor_pos + 1
                    neighbor_list[neighbor_pos] = [seedn, self.target_image[seedn]]
                    tmp_image[seedn] = 1

            tmp_image[seed] = 2
            self.inner_image[seed] = self.target_image[seed]
            region_size += 1

            distance = np.abs(neighbor_list[:neighbor_pos + 1, len(image_shape)] - np.tile(origin_t, neighbor_pos + 1))
            index = distance.argmin()
            seed = neighbor_list[index][:len(image_shape)]
            neighbor_list[index] = neighbor_list[neighbor_pos]
            neighbor_pos -= 1
        return self.inner_image


class fixed_region_grow(RegionGrowing):
    """
    Fixed threshold region growing.
    """
    def __init__(self, target_image, seed, Thres):
        if not isinstance(seed,np.ndarray):
            seed = np.array(seed)
        self.target_image = target_image
        self.set_seed(seed)
        self.set_stop_criteria(Thres)

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

    def _grow(self, image, seed, Num):
        """
        Average contrast growing.
        """
        #N = self.get_stop_criteria().value
        return self.grow(image, seed, Num)


class Average_contrast(RegionGrowing):
    """
    Max average contrast region growing.
    """
    def __init__(self, target_image, seed, Thres, connectivity):
        if not isinstance(seed,np.ndarray):
            seed = np.array(seed)
        self.target_image = target_image
        self.set_seed(seed)
        self.get_seed()
        self.get_thres = Thres
        self.set_connectivity(connectivity)
        self.get_connectivity()

    def set_seed(self, seed):
        """
        Set the seed points.
        """
        self.seed = seed

    def get_seed(self):
        """
        Return the seed points.
        """
        return self.seed

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
            for i in range(self.get_connectivity()):
                set0,set1,set2 = compute_offsets(len(image.shape),self.get_connectivity())[i]
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
            x,y,z = neighbor_list[index][:len(image.shape)]
            inner_list = inner_list + [image[x,y,z]]
            neighbor_list[index] = neighbor_list[neighbor_pos]
            neighbor_pos -= 1
        number = int(np.array(contrast).argmax()+1)
        print number
        self.stop_criteria = StopCriteria('size',number)

    def get_stop_criteria(self):
        """
        Return the stop criteria.
        """
        return self.stop_criteria

    def grow(self):
        """
        Average contrast growing.
        """
        seed = self.get_seed()
        image_shape = self.target_image.shape
        image = self.target_image
        Num = self.get_thres
        self.set_stop_criteria(image, seed, Num)
        N = self.get_stop_criteria()

        region_size = 1
        origin_t = self.target_image[seed]
        tmp_image = np.zeros_like(self.target_image)
        self.inner_image = np.zeros_like(self.target_image)

        neighbor_free = 10000
        neighbor_pos = -1
        neighbor_list = np.zeros((neighbor_free, len(image.shape)+1))

        while region_size <= N:
            for i in range(self.get_connectivity()):
                set0,set1,set2 = compute_offsets(len(image.shape),self.get_connectivity())[i]
                xn,yn,zn = seed[0]+set0,seed[1]+set1,seed[2]+set2
                if inside((xn,yn,zn),image_shape) and tmp_image[xn,yn,zn]==0:
                    neighbor_pos = neighbor_pos + 1
                    neighbor_list[neighbor_pos] = [xn,yn,zn,image[xn,yn,zn]]
                    tmp_image[xn,yn,zn] = 1

            tmp_image[seed] = 2
            self.inner_image[seed] = self.target_image[seed]
            region_size += 1

            distance = np.abs(neighbor_list[:neighbor_pos + 1, len(image_shape)] - np.tile(origin_t, neighbor_pos + 1))
            index = distance.argmin()
            seed = neighbor_list[index][:len(image_shape)]
            neighbor_list[index] = neighbor_list[neighbor_pos]
            neighbor_pos -= 1
        return self.inner_image


class Peripheral_contrast(RegionGrowing):
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
    A = Average_contrast(data, (26,38,25), 1000, 26)
    new_image = A.grow()
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



























