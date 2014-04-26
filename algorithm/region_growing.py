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
        seed: the sedd points.
        stop_criteria: The stop criteria of region growing to stop.
        """
        if not isinstance(target_image, np.ndarray):
            img = nib.load(target_image)
            if len(img.shape) > 4 or len(img.shape) <2:
                raise ValueError("Must be a 2D/3D/4D Nifti1 format file.")
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
        return self.stop_criteria

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




















