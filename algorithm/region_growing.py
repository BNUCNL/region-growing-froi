import numpy as np
import nibabel as nib


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

        if not isinstance(seed, list):
            seed = [seed]

    def set_seed(self, seed):
        """
        Set the seed points.
        """
        self.set_seed()

    def get_seed(self):
        """
        Return the seed points.
        """
        return self.get_seed()

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

    def set_connectivity(self):
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

    def inside(coordinate,image_shape):
        """
        whether the coordinate is in the image,return True or False.
        """
        return  (coordinate[0] >= 0) and (coordinate[0] < image_shape[0]) and \
                (coordinate[1] >= 0) and (coordinate[1] < image_shape[1]) and \
                (coordinate[2] >= 0) and (coordinate[2] < image_shape[2])

    def grow(self):
        """
        The main region grow function.
        """
        return self.grow()

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

        if not isinstance(value, float) or not isinstance(value, int):
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
    def __init__(self, target_image, seed, value):
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


# main:
# a = FixedThresholdSRG(x,y,z)
# ret = a.grow()


























