# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import time
import nibabel as nib
import numpy as np

from algorithm.region_growing import SeededRegionGrowing

if __name__ == "__main__":
    starttime = time.clock()
    #load data
    image = nib.load("../data/S1/zstat1.nii.gz")
    affine = image.get_affine()
    image = image.get_data()

    # init r_FFA seed point: [25, 41, 25]
    seed_coords = np.array([[25, 41, 25]])
    srg_object = SeededRegionGrowing(seed_coords, 500)
    # srg_object.convert_image_to_regions(image, brain_mask)

    r_FFA_mask = nib.load("../data/prior/prob_rFFA.nii.gz").get_data().astype(np.bool)
    result_region = srg_object.compute(image, r_FFA_mask)

    #Convert the region to image
    result_image = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
    label_mask = result_region.generate_region_mask()
    result_image[label_mask] = 1

    #save the result image to disk
    nib.save(nib.Nifti1Image(result_image, affine), "../data/S1/new_srg_zstat1_srg.nii.gz")

    endtime = time.clock()
    print 'Cost time:: ', np.round((endtime - starttime), 3), ' s'
    #It costs about 71s without the whole brain mask while 14s with the whole brain mask