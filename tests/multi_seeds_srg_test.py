__author__ = 'zgf'

import time
import nibabel as nib

from algorithm.region_growing import *
from algorithm.similarity_criteria import *
from algorithm.stop_criteria import *
from algorithm.region import *

if __name__ == "__main__":
    #load data
    starttime = time.clock()
    image = nib.load("../data/S1/zstat1.nii.gz")
    affine = image.get_affine()
    image = image.get_data()

    # subject_id		r_OFA			l_OFA			r_pFus			l_pFus
    # S1			(28, 29, 29)	(63, 26, 28)	(25, 41, 25)	(65, 42, 26)
    r_OFA_seed_coords = np.array([[28, 29, 29]])
    l_OFA_seed_coords = np.array([[63, 26, 28]])
    r_pFus_seed_coords = np.array([[25, 41, 25]])
    l_pFus_seed_coords = np.array([[65, 42, 26]])
    seed_coords = [r_OFA_seed_coords, l_OFA_seed_coords, r_pFus_seed_coords, l_pFus_seed_coords]

    neighbor_element = SpatialNeighbor('connected', image.shape, 26)

    r_OFA_region = Region(r_OFA_seed_coords, neighbor_element)
    l_OFA_region = Region(l_OFA_seed_coords, neighbor_element)
    r_pFus_region = Region(r_pFus_seed_coords, neighbor_element)
    l_pFus_region = Region(l_pFus_seed_coords, neighbor_element)
    regions = [r_OFA_region, l_OFA_region, r_pFus_region, l_pFus_region]



    similarity_criteria = SimilarityCriteria('euclidean')
    stop_criteria = StopCriteria('size')
    threshold = np.array((1600, ))



    multi_seeds_srg = MultiSeedsSRG(similarity_criteria, stop_criteria)
    srg_region = multi_seeds_srg.compute(regions, image, threshold)


    result_image = np.zeros((image.shape[0], image.shape[1], image.shape[2], len(srg_region)))
    for i in range(len(srg_region)):
        labels = srg_region[i].get_label()
        result_image[labels[:, 0], labels[:, 1], labels[:, 2], i] = 1

    nib.save(nib.Nifti1Image(result_image, affine), "../data/S1/zstat1_srg.nii.gz")

    endtime = time.clock()
    print 'Cost time:: ', np.round((endtime - starttime), 3), ' s'
