# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import time
import nibabel as nib

from algorithm.unsed.region_growing import *
from algorithm.unsed.neighbor import *
from algorithm.unsed.similarity_criteria import *
from algorithm.unsed.stop_criteria import *
from algorithm.unsed.region import *

if __name__ == "__main__":
    starttime = time.clock()

    # image to be segmanted
    image = nib.load("../data/S1/zstat1.nii.gz")
    affine = image.get_affine()
    image = image.get_data()

    # load prior image
    prior = nib.load("../data/prior/prob_rFFA.nii.gz")
    prior_image = prior.get_data()
    # get the prior coords
    seed_coords = np.array(np.nonzero(prior_image >= 0.6)).T
    #init the SpatialNeighbor object
    neighbor_element = SpatialNeighbor('connected', prior_image.shape, 26)
    #init the region object
    region = Region(seed_coords, neighbor_element)

    # init the PriorBasedSimilarityCriteria object
    similarity_criteria = PriorBasedSimilarityCriteria(prior_image, 'DB', 0.5)
    #int the StopCriteria object
    stop_criteria = StopCriteria('size')

    # init the threshold
    threshold = np.array((15, 30))

    # init the SeededRegionGrowing object
    prior_srg = SeededRegionGrowing(similarity_criteria, stop_criteria)
    #compute the regions
    prior_srg_region = prior_srg.compute(region, image, threshold)

    #Convert the region to image
    result_image = np.zeros((image.shape[0], image.shape[1], image.shape[2], len(prior_srg_region)))
    for i in range(len(prior_srg_region)):
        labels = prior_srg_region[i].get_label()
        result_image[labels[:, 0], labels[:, 1], labels[:, 2], i] = 1
    
    #save the result image to disk
    nib.save(nib.Nifti1Image(result_image, affine), "../data/S1/zstat1_prior_srg.nii.gz")


