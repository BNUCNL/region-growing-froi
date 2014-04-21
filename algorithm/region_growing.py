import numpy as np
import nibabel as nib

class RegionGrowing:
   """
   Base class in region growing.

   """
    def __init__(self, input, seed, similarity_criteria, stop_criteria, output):
        """
         Parameters
        ----------
        input: must be 2D/3D/4D np.ndarray type or a Nifti1 format file(*.nii, *.nii.gz).
        seed: the sedd points.
        stop_criteria: The stop criteria of region growing to stop.
        output: must be 2D/3D/4D np.ndarray type or the path string of a Nifti1 format file(*.nii, *.nii.gz)
        """
        if not isinstance(input, np.ndarray):
            img = nib.load(input)
            if len(img.shape) > 4 or len(img.shape) <2:
                raise ValueError("Must be a 2D/3D/4D Nifti1 format file.")
        elif len(input.shape) > 4 or len(input.shape) < 2:
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
        return self.growing()


class ThreeOrTwoDRegionGrowing(RegionGrowing):
    def __init__(self, input, seed, stop_criteria, output):
        RegionGrowing.__init__(self)
        self.set_seed(seed)
        self.data = input
        self.output = output
        self.set_stop_criteria(stop_criteria)

    def set_stop_criteria(self, stop_criteria):
        self.stop_criteria = stop_criteria

    def get_stop_criteria(self):
        return self.stop_criteria

    def getVoxel(self, x, y, z, data):                  #get the value of Voxel point
        if x < 0 or x > data.shape[0]:
            return False

        if y < 0 or y > data.shape[1]:
            return False

        if z < 0 or z > data.shape[2]:
            return False

        return data[x,y,z]

    def setVoxel(x, y, z, data, value):           #set the value of Voxel point
        if x < 0 or x > data.shape[0]:
            return False

        if y < 0 or y > data.shape[1]:
            return False

        if z < 0 or z > data.shape[2]:
            return False

        data[x, y, z] = value
        return True

    def growing(self):
    #define function(original image\gradient difference\start point)radient

        Q = Queue()                                      #Q is the queue of flag 1
        s = []                                           #s is the list of flag 2
        #flag=0

        x = self.seed[0]
        y = self.seed[1]
        z = self.seed[2]

        Q.enque((x,y,z))


        while not Q.isEmpty():
            #the circle when it is not empty
            t = Q.deque()
            x = t[0]
            y = t[1]
            z = t[2]
            shape = self.data.shape

            if x < shape[0] and abs(self.getVoxel(x+1, y, z, self.data) - self.getVoxel(x, y, z, self.data)) <= self.stop_criteria :

                if not Q.isInside((x + 1, y, z)) and not (x + 1, y, z) in s:
                #the point is not in the Q and s
                    Q.enque((x + 1, y, z))
                    #then insert the point


            if x > 0 and abs(self.getVoxel(x - 1, y, z, self.data) - self.getVoxel(x, y, z,self.data)) <= self.stop_criteria:

                if not Q.isInside((x - 1, y, z)) and not (x - 1, y, z) in s:
                    Q.enque( (x - 1, y, z))


            if y < shape[1] and abs(self.getVoxel(x, y+1, z, self.data) - self.getVoxel(x, y, z, self.data)) <= self.stop_criteria:

                if not Q.isInside((x, y + 1, z) ) and not (x, y + 1, z) in s:
                    Q.enque( (x, y + 1, z))


            if y > 0 and abs(self.getVoxel(x, y - 1, z, self.data) - self.getVoxel(x, y, z, self.data)) <= self.stop_criteria:

                if not Q.isInside((x, y - 1, z)) and not (x, y - 1, z) in s:
                    Q.enque((x, y - 1, z))

            if z < shape[2] and abs(self.getVoxel(x, y, z + 1, self.data) - self.getVoxel(x, y, z, self.data)) <= self.stop_criteria:

                if not Q.isInside((x, y, z + 1)) and not (x, y, z + 1) in s:
                    Q.enque( (x, y , z+1) )


            if z > 0 and abs(self.getVoxel(x, y, z - 1, self.data) - self.getVoxel(x, y, z, self.data)) <= self.stop_criteria:

                if not Q.isInside((x, y, z - 1)) and not (x, y, z - 1) in s:
                    Q.enque((x, y, z - 1))


            if t not in s:
                s.append(t)             #affix the start point
                #flag=flag+1


        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                for k in range(0, shape[2]):
                    if not (i, j, k) in s:
                        self.data[x, y, z] = 0

























