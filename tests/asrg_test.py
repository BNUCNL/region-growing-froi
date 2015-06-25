# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import time
import nibabel as nib

from algorithm.region_growing import *
from algorithm.neighbor import *
from algorithm.similarity_criteria import *
from algorithm.stop_criteria import *
from algorithm.region import *
from algorithm.optimizer import *

if __name__ == "__main__":
    #load data
    starttime = time.clock()
    #load test data
    image = nib.load("../data/S1/zstat1.nii.gz")
    affine = image.get_affine()
    image = image.get_data()

    #init the seed points
    seed_coords = np.array([[28, 29, 29]]) # r_OFA seed point: [28, 29, 29]
    #init the SpatialNeighbor object
    neighbor_element = SpatialNeighbor('connected', image.shape, 26)
    #init the region object
    region = Region(seed_coords, neighbor_element)

    #init the SimilarityCriteria object
    similarity_criteria = SimilarityCriteria('euclidean')
    #init the StopCriteria object
    stop_criteria = StopCriteria('size')
    threshold = np.array((50, 100, 150, 200))

    #init the SeededRegionGrowing object
    srg = SeededRegionGrowing(similarity_criteria, stop_criteria)
    #compute the regions
    srg_region = srg.compute(region, image, threshold)

    #Convert the region to image
    result_image = np.zeros((image.shape[0], image.shape[1], image.shape[2], len(srg_region)))
    for i in range(len(srg_region)):
        labels = srg_region[i].get_label()
        result_image[labels[:, 0], labels[:, 1], labels[:, 2], i] = 1
    #save the result image to disk
    nib.save(nib.Nifti1Image(result_image, affine), "../data/S1/zstat1_srg.nii.gz")

    #init AC optimizer object
    optimizer_AC = Optimizer('AC')
    optimizer_AC_image = optimizer_AC.compute([srg_region], image)
    #init PC optimizer object
    # optimizer_PC = Optimizer('PC')
    # optimier_PC_image = optimizer_PC.compute(rsrg_region, image[..., j])

    #get the optimizer region index
    index = optimizer_AC_image[0].argmax()
    optimal_regions = srg_region[index]
    labels = optimal_regions.get_label()

    result_image = np.zeros_like(image)
    result_image[labels[:, 0], labels[:, 1], labels[:, 2]] = 1
    #save the result image to disk
    nib.save(nib.Nifti1Image(result_image, affine), "../data/S1/zstat1_asrg_images.nii.gz")

    endtime = time.clock()
    print 'Cost time:: ', np.round((endtime - starttime), 3), ' s'
