import time

import nibabel as nib

from algorithm.regiongrowing import *
from algorithm.neighbor import *

if __name__ == "__main__":
    mask = nib.load("../data/prior/prob_rFFA.nii.gz")
    mask = mask.get_data()

    seed_coords = np.array(np.nonzero(mask >= 0.6)).T
    neighbor_element = SpatialNeighbor('connected', mask.shape, 26)
    region = Region(seed_coords, neighbor_element)

    similarity_criteria = SimilarityCriteria('euclidean', 0.8)
    stop_criteria = StopCriteria('size')

    image = nib.load("../data/S2/tstat1.nii.gz")
    affine = image.get_affine()
    image = image.get_data()

    threshold = np.array((3, 5))

    starttime = time.clock()

    # Test for SRG
    srg = SeededRegionGrowing(similarity_criteria, stop_criteria)
    srg_region = srg.compute(region, image, threshold)
    for i in range(len(srg_region)):
        print i, srg_region[i].label

        #
        # endtime = time.clock()
        #
        # print(endtime - starttime)




        #print region.shape

        #aggregator = Aggregator('MWA')
        #srg_image = aggregator.compute(region, image)


        #srg_image = np.zeros_like(image, dtype=int)
        #srg_image[region_label[:, 0], region_label[:, 1], region_label[:, 2]] = 1
        #nib.save(nib.Nifti1Image(srg_image, affine), "../data/S2/RSRG3d.nii.gz")

