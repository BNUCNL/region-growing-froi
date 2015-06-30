# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import time
import nibabel as nib

from algorithm.unsed.region_growing import *
from algorithm.unsed.similarity_criteria import *
from algorithm.unsed.stop_criteria import *
from algorithm.unsed.region import *

if __name__ == "__main__":
    starttime = time.clock()
    #load data
    image = nib.load("../data/S1/zstat1.nii.gz")
    affine = image.get_affine()
    image = image.get_data()

    # init r_OFA seed point: [28, 29, 29]
    seed_coords = np.array([[28, 29, 29]]) 
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

    endtime = time.clock()
    print 'Cost time:: ', np.round((endtime - starttime), 3), ' s'
