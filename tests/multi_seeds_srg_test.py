# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import time
import nibabel as nib

from algorithm.region_growing import *
from algorithm.similarity_criteria import *
from algorithm.stop_criteria import *
from algorithm.region import *

if __name__ == "__main__":
    starttime = time.clock()
    #load data
    image = nib.load("../data/S1/zstat1.nii.gz")
    affine = image.get_affine()
    image = image.get_data()
    
    #roi seed point (reoi peak point) information
    # subject_id		r_OFA			l_OFA			r_pFus			l_pFus
    # S1			(28, 29, 29)	(63, 26, 28)	(25, 41, 25)	(65, 42, 26)
    r_OFA_seed_coords = np.array([[28, 29, 29]])
    l_OFA_seed_coords = np.array([[63, 26, 28]])
    r_pFus_seed_coords = np.array([[25, 41, 25]])
    l_pFus_seed_coords = np.array([[65, 42, 26]])
 
    #init the SpatialNeighbor object
    neighbor_element = SpatialNeighbor('connected', image.shape, 26)

    #init the regions corresponding the multi seeds, e.g. r_OFA, l_OFA, r_pFus, l_pFus
    r_OFA_region = Region(r_OFA_seed_coords, neighbor_element)
    l_OFA_region = Region(l_OFA_seed_coords, neighbor_element)
    r_pFus_region = Region(r_pFus_seed_coords, neighbor_element)
    l_pFus_region = Region(l_pFus_seed_coords, neighbor_element)
    regions = [r_OFA_region, l_OFA_region, r_pFus_region, l_pFus_region]
    
    #init the MultiSeedsStopCriteria object.
    multi_seeds_stop_criteria = MultiSeedsStopCriteria('size')
    threshold = np.array((100, 300, 600, 900, 1200))

    #init the MultiSeedsSimilarityCriteria object
    multi_seeds_similarity_criteria = MultiSeedsSimilarityCriteria('euclidean')

    #init the MultiSeedsSRG object
    multi_seeds_srg = MultiSeedsSRG(multi_seeds_similarity_criteria, multi_seeds_stop_criteria)
    #compute the multi seeds region
    multi_seeds_srg_region = multi_seeds_srg.compute(regions, image, threshold, neighbor_element)

    #Convert the region to image
    result_image = np.zeros((image.shape[0], image.shape[1], image.shape[2], len(multi_seeds_srg_region)))
    for i in range(len(multi_seeds_srg_region)):
        for j in range(len(multi_seeds_srg_region[i])):
            roi_label = multi_seeds_srg_region[i][j].get_label()
            result_image[roi_label[:, 0], roi_label[:, 1], roi_label[:, 2], i] = j + 1
    #save the result image to disk
    nib.save(nib.Nifti1Image(result_image, affine), "../data/S1/zstat1_multi_seeds_srg.nii.gz")

    endtime = time.clock()
    print 'Cost time:: ', np.round((endtime - starttime), 3), ' s'























