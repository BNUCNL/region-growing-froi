import os
import time
import nibabel as nib

from algorithm.regiongrowing import *
from algorithm.neighbor import *


if __name__ == "__main__":
    # image to be segmanted
    image = nib.load("../data/S2/tstat1.nii.gz")
    data = image.get_data()

    # prior image and seed region
    prior = nib.load("../data/prior/prob_rFFA.nii.gz")
    prior_data = prior.get_data()
    starttime = time.clock()
    seed_coords = np.array(np.nonzero(prior_data >= 0.6)).T
    neighbor_element = SpatialNeighbor('connected', prior_data.shape, 26)
    region = Region(seed_coords, neighbor_element)

    # similarity
    similarity_criteria = PriorBasedSimilarityCriteria(prior_data)
    #similarity_criteria = SimilarityCriteria()
    stop_criteria = StopCriteria('size')

    # region size
    threshold = np.array((100, 200, 300, 400, 500, 600))

    # Test for SRG
    srg = SeededRegionGrowing(similarity_criteria, stop_criteria)
    regions = []
    srg_region = srg.compute(region, data, threshold)
    regions.append(srg_region)
    for i in range(len(srg_region)):
        print i, srg_region[i].label.shape

    #test optimizer******************************************ok
    optimizer = Optimizer('AC')
    #optimizer = Optimizer('PC')
    optimizer_region = optimizer.compute(regions, data)
    print 'optimal threshold: %d' % threshold[optimizer_region.reshape(len(threshold),).argmax()]
    AC_labels = srg_region[optimizer_region.reshape(len(threshold),).argmax()].label
    srg_data = np.zeros_like(data, dtype=int)
    srg_data[AC_labels[:, 0], AC_labels[:, 1], AC_labels[:, 2]] = 1
    image._data = srg_data
    nib.save(image, "../data/S2/AC_SRG3d.nii.gz")
    #nib.save(image, "../data/S2/PC_SRG3d.nii.gz")

    endtime = time.clock()
    print 'Time: %.2f' % (endtime - starttime)
    print 'niftil image has saved'
