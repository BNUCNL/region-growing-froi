__author__ = 'zhenzonglei'

import time
import nibabel as nib

from algorithm.region_growing import *
from algorithm.neighbor import *
from algorithm.similarity_criteria import *
from algorithm.stop_criteria import *
from algorithm.region import *

if __name__ == "__main__":
    # image to be segmanted
    image = nib.load("../data/S1/tstat1.nii.gz")
    affine = image.get_affine()
    image = image.get_data()

    # prior image and seed region
    prior = nib.load("../data/prior/prob_rFFA.nii.gz")
    prior_image = prior.get_data()
    seed_coords = np.array(np.nonzero(prior_image >= 0.6)).T
    neighbor_element = SpatialNeighbor('connected', prior_image.shape, 26)
    region = Region(seed_coords, neighbor_element)

    # similarity
    similarity_criteria = PriorBasedSimilarityCriteria(prior_image, 'DB', 0.5)
    stop_criteria = StopCriteria('size')

    # region size
    threshold = np.array((15, 30))
    starttime = time.clock()

    # Test for priorSRG
    srg = SeededRegionGrowing(similarity_criteria, stop_criteria)
    srg_region = srg.compute(region, image, threshold)
    for i in range(len(srg_region)):
        print i, srg_region[i].label.shape

    endtime = time.clock()
    print(endtime - starttime)


    #print region.shape

    #aggregator = Aggregator('MWA')
    #srg_image = aggregator.compute(region, image)


    #srg_image = np.zeros_like(image, dtype=int)
    #srg_image[region_label[:, 0], region_label[:, 1], region_label[:, 2]] = 1
    #nib.save(nib.Nifti1Image(srg_image, affine), "../data/S2/RSRG3d.nii.gz")

