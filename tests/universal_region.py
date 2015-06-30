# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import time
import nibabel as nib

from algorithm.unsed.region_growing import *

if __name__ == "__main__":
    starttime = time.clock()
    #load data
    image = nib.load("../data/S1/zstat1.nii.gz")
    affine = image.get_affine()
    image = image.get_data()

    # init r_OFA seed point: [28, 29, 29]
    seed_coords = np.array([[28, 29, 29]])

    #generate the unique value for every voxel
    new__image = np.arange(image.shape[0] * image.shape[1] * image.shape[2])[:].reshape(image.shape) + 1
    print new__image






    #mapping the image to primary region
    regions = []



    endtime = time.clock()
    print 'Cost time:: ', np.round((endtime - starttime), 3), ' s'
