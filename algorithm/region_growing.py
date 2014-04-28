import numpy as np
import nibabel as nib
from connectivity import compute_offsets
from whetherinside import inside


class RegionGrowing:
    """
    Base class in region growing.

    """
    def __init__(self, target_image, seed, stop_criteria, connectivity='8', similarity_criteria='euclidean', mask_image=None, prior_image=None):
        """
         Parameters
        ----------
        input: must be 2D/3D/4D np.ndarray type or a Nifti1 format file(*.nii, *.nii.gz).
        seed: the seed points.
        stop_criteria: The stop criteria of region growing to stop.
        """
        if not isinstance(target_image, nib.nifti1.Nifti1Image):
            if len(target_image.get_shape()) > 4 or len(target_image.get_shape()) <2:
                raise ValueError("Must be Nifti1Image format file.")
        elif len(target_image.shape) > 4 or len(target_image.shape) < 2:
            raise ValueError("Must be a 2D/3D/4D data.")

        if not isinstance(seed,np.ndarray):
            seed = np.array(seed)

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
        return self.set_connectivity()

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

    def grow(self, image, seed, Num):
        """
        Give a coordinate ,return a region.
        """
        x,y,z = seed
        image_shape = image.shape

        if inside(seed,image_shape)!=True:
            print "The seed is out of the image range."
            return False

        region_size = 1
        origin_t = image[x,y,z]
        tmp_image = np.zeros_like(image)
        self.inner_image = np.zeros_like(image)

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
            self.inner_image[x,y,z] = image[x,y,z]
            region_size += 1

            distance = np.abs(neighbor_list[:neighbor_pos+1,3] - np.tile(origin_t,neighbor_pos+1))
            index = distance.argmin()
            x,y,z = neighbor_list[index][:3]
            neighbor_list[index] = neighbor_list[neighbor_pos]
            neighbor_pos -= 1

        return self.inner_image

class SimilarityCriteria:
    """
    Distance measure.
    """

    def __init__(self, X, name='euclidean', similarity_direction='seed'):
        """
        Parameters
        -----------------------------------------------------
        X: A matrix contain m n-dimensional row vectors.
        name: 'educlidean', 'mahalanobis', 'minkowski','seuclidean', 'cityblock',ect.
        similarity_direction: 'seed', 'neighbor', 'mutual'
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("The input X matrxi must be np.ndarray type. ")

        if not isinstance(name, str):
            raise ValueError("The value of name must be str type. ")
        else:
            self.set_name(name)
        if not isinstance(similarity_direction, str):
            raise ValueError("The value of similarity direction must be str type. ")
        else:
            self.set_similarity_direction(similarity_direction)

    def set_name(self, name):
        """
        Set the name of the distances which to use.
        """
        self.name = name

    def get_name(self):
        """
        Get the name of the distance used in the region growing.
        """
        return self.name

    def set_similarity_direction(self, similarity_direction):
        """
        Set the similarity direction.
        """
        from scipy.spatial.distance import pdist

        self.similarity_direction = pdist(self.X, similarity_direction)

    def get_similarity_direction(self):
        """
        Get the similarity direction.
        """
        return self.similarity_direction

class StopCriteria:
    """
    Stop criteria.
    """
    def __init__(self, name, mode, value):
        """
        Parameters
        -----------------------------------------------------
        name: 'size' or 'homogeneity',  means the size or the homogeneity of the region.
        mode: 'fixed' or 'adaptive', means the threshold should be fixed value or adaptive.
        value: fixed value, it should be none when mode is equal to 'adaptive'.
        """

        if not isinstance(name, str):
            raise ValueError("The name must be str type. ")
        elif name is not 'size' and name is not 'homogeneity':
            raise ValueError("The name must be 'size' or 'homogeneity'.")
        else:
            self.set_name(name)

        if not isinstance(mode, str):
            raise ValueError("The mode must be str type. ")
        elif mode is not 'fixed' and mode is not 'adaptive':
            raise ValueError("The mode must be 'fixed' or 'adaptive'.")
        else:
            self.set_mode(mode)

        if not isinstance(value, float) and not isinstance(value, int):
            raise ValueError("The value must be float or int type.")
        else:
            self.set_value(value)

    def set_name(self, name):
        """
        Set the name of the stop criteria.
        """
        self.name = name

    def get_name(self):
        """
        Get the name of the stop criteria.
        """
        return self.name

    def set_mode(self, mode):
        """
        Set the mode of the stop criteria.
        """
        self.mode = mode

    def get_mode(self):
        """
        Get the mode of the stop criteria.
        """
        return self.mode

    def set_value(self, value):
        """
        Set the value of the stop criteria.
        """
        if self.get_mode() == 'adaptive':
            self.value = None
        else:
            self.value = value

    def get_value(self, value):
        """
        Get the value of the stop criteria.
        """
        return self.value

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
            if len(target_image.get_shape()) > 3 or len(target_image.get_shape()) <2:
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
        self.connectivity = connectivity

    def get_connectivity(self):
        """
        Get the connectivity.
        """
        return self.get_connectivity()

    def grow(self):
        """
        Fixed threshold region growing.
        """
        Q = []
        #list of new picture point
        s = []

        if len(self.target_image.shape) == 2:
            #(x,y) start point
            x, y = self.seed

            #the function to transfer image to grey-scale map
            image = self.target_image.convert("L")
            Q.insert((x, y))

            while len(Q) != 0:
                t = Q.pop()
                x = t[0]
                y = t[1]
                if self.get_connectivity() == '6':
                    #in the size of picture and the gradient difference is not so large

                    if x < image.size[0] - 1 and \
                            abs(image.getpixel((x + 1, y)) - image.getpixel((x, y))) <= self.stop_criteria.get_value():

                        if not (x + 1, y) in Q and not (x + 1, y) in s:
                            Q.insert((x + 1, y))


                    if x > 0 and \
                            abs(image.getpixel((x - 1, y)) - image.getpixel((x, y))) <= self.stop_criteria.get_value():

                        if not (x - 1, y) in Q and not (x - 1, y) in s:
                            Q.insert((x - 1, y))

                    if y < (image.size[1] - 1) and \
                            abs(image.getpixel((x, y + 1)) - image.getpixel((x, y))) <= self.stop_criteria.get_value():

                        if not (x, y + 1) in Q and not (x, y + 1) in s:
                            Q.insert((x, y + 1))

                    if y > 0 and \
                            abs( image.getpixel((x, y - 1)) - image.getpixel((x, y))) <= self.stop_criteria.get_value():
                        if not (x, y - 1) in Q and not (x, y - 1) in s:
                            Q.insert((x, y - 1))

                    if t not in s:
                        s.append(t)

                image.load()
                putpixel = image.im.putpixel

                for i in range(image.size[0]):
                    for j in range(image.size[1]):
                        putpixel( (i, j), 0)

                for i in s:
                    putpixel(i, 150)
                return image
        elif len(self.target_image.shape) == 3:
            #define function(original image\gradient difference\start point)radient
            Q = []
            s = []

            x, y, z = self.seed
            Q.insert((x, y, z))

            while len(Q) != 0:
                t = Q.pop()
                x = t[0]
                y = t[1]
                z = t[2]

                if self.get_connectivity() == '6':
                    if x < self.target_image.shape[0] and \
                            abs(self.target_image[x + 1, y, z] - self.target_image[x, y, z]) <= self.stop_criteria.get_value():
                        if not (x + 1, y, z)in Q and not (x + 1, y, z) in s:
                            Q.insert((x + 1, y, z))

                    if x > 0 and abs(self.target_image[x - 1, y, z] -
                        self.target_image[x, y, z]) <= self.stop_criteria.get_value():
                        if not (x - 1, y, z) in Q and not (x - 1, y, z) in s:
                            Q.insert((x - 1, y, z))

                    if y < self.target_image.shape[1] and \
                            abs(self.target_image[x, y + 1, z] - self.target_image[x, y, z]) <= self.stop_criteria.get_value():
                        if not (x, y + 1,z) in Q and not (x, y + 1, z) in s:
                            Q.insert((x, y + 1, z))

                    if y > 0 and \
                            abs(self.target_image[x, y - 1, z] - self.target_image[x, y, z]) <= self.stop_criteria.get_value():
                        if not (x, y - 1 ,z) in Q and not (x, y - 1, z) in s:
                            Q.insert((x, y - 1, z))

                    if z < self.target_image.shape[2] and \
                            abs(self.target_image[x, y, z + 1] - self.target_image[x, y, z]) <= self.stop_criteria.get_value():
                        if not (x, y, z + 1) in Q and not (x, y, z + 1) in s:
                            Q.insert((x, y, z + 1))

                    if z > 0 and \
                            abs(self.target_image[x, y, z - 1] - self.target_image[x, y, z]) <= self.stop_criteria.get_value():
                        if not (x, y, z - 1) in Q and not (x, y, z - 1) in s:
                            Q.insert((x, y, z - 1))
                    if t not in s:
                        s.append(t)

                array = np.array(s).transpose()
                self.output = self.target_image.copy()
                self.output[array[:][0], array[:][1], array[:][2]] = 1

                return self.output

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
                    neighbor_pos = neighbor_pos+1
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
        self.stop_criteria = StopCriteria('size','fixed',number)

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
    C = fixed_region_grow(data,(26,38,25),200)
    new_image = C._grow(data,(26,38,25),200)
    t_image._data = new_image
    nib.save(t_image,'fixed_thres_S2_image.nii.gz')
    print 'fixed threshold growing has been saved.'



























