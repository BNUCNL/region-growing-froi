import time
import sys

import nibabel as nib
import numpy as np

from algorithm.regiongrowing import *
from algorithm.neighbor import *

RESULT_DIR = "/nfs/j3/userhome/zhouguangfu/workingdir/SRG/result/"


import os
import numpy as np
import nibabel as nib
import csv
from scipy.spatial.distance import cdist

def hausdorff(vector1, vector2):
    D = cdist(vector1, vector2, 'euclidean')

    #none symmetric Hausdorff distances
    H1 = np.max(np.min(D, axis=1))
    H2 = np.max(np.min(D, axis=0))

    return (H1 + H2) / 2.




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

    threshold = np.array((200, ))

    srg = SeededRegionGrowing(similarity_criteria, stop_criteria)
    srg_region = srg.compute(region, image, threshold)
    nib.save(nib.Nifti1Image(srg_region, affine), RESULT_DIR + "SRG3d.nii.gz")
    #
    #
    # for i in range(len(srg_region)):
    #          print i,srg_region[i].label.shape[0]


    starttime = time.clock()
    rand_neighbor_prop = 0.7

    # for i in range(1, 2):
    #     seed_sampling_num = i * 5
    #     similarity_criteria.set_rand_neighbor_prop(0.7)
    #     rsrg = RandomSRG(similarity_criteria, stop_criteria, seed_sampling_num)
    #     rsrg_region = rsrg.compute(region, image, threshold)
    #
    #     aggregator = Aggregator('MWA')
    #     rsrg_image = aggregator.compute(rsrg_region, image)
    #     nib.save(nib.Nifti1Image(rsrg_image, affine), RESULT_DIR + "RSRG3d.nii.gz")
    #
    # endtime = time.clock()
    # print str(endtime - starttime) + 's'




    print "Program end..."

