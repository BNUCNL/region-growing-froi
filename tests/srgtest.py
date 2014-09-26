import time
import sys
import os
import numpy as np
import nibabel as nib
import csv

from algorithm.regiongrowing import *
from algorithm.neighbor import *

SUBJECT_ID_DIR = "G:/nfs/4Ddata/2006subID"
RESULT_DIR = "G:/workingdir/result/"
ACTIVATION_DATA_DIR = "G:/nfs/t2/atlas/group/face-object/activation/2006zstat.nii.gz"
FOUR_D_DATA_DIR = "G:/nfs/4Ddata/"
RESULT_NPY_FILE = "peak_points_all_sub.npy"
ROI = ['r_OFA', 'l_OFA', 'r_pFus', 'l_pFus']


if __name__ == "__main__":
    mask = nib.load("../data/prior/prob_rFFA.nii.gz")
    mask = mask.get_data()

    # # seed_coords = np.array(np.nonzero(mask >= 0.6)).T
    # seed_coords = np.array([[31, 30, 8]])
    #
    # neighbor_element = SpatialNeighbor('connected', mask.shape, 26)
    # region = Region(seed_coords, neighbor_element)


    roi_peak_points = np.load(FOUR_D_DATA_DIR + RESULT_NPY_FILE)


    # similarity_criteria = SimilarityCriteria('euclidean', 0.8)
    similarity_criteria = SimilarityCriteria('euclidean')
    stop_criteria = StopCriteria('size')

    image = nib.load(ACTIVATION_DATA_DIR)
    affine = image.get_affine()
    image = image.get_data()

    threshold = np.array((300, ))
    srg = SeededRegionGrowing(similarity_criteria, stop_criteria)

    for i in range(len(ROI)):
        region_image = np.zeros((image.shape[0], image.shape[1], image.shape[2], image.shape[3]), dtype=int)
        for j in range(0, image.shape[3]):
            seed_coords = np.array([roi_peak_points[j, i, :]]).astype(np.int)
            neighbor_element = SpatialNeighbor('connected', mask.shape, 26)
            region = Region(seed_coords, neighbor_element)

            srg_region = srg.compute(region, image[..., j], threshold)
            labels = srg_region[0].get_label()
            region_image[labels[:, 0], labels[:, 1], labels[:, 2], j] = 1
            print 'i-----', i, ' j ----- ', j, '----label size: ', srg_region[0].label_size()

        nib.save(nib.Nifti1Image(region_image, affine), RESULT_DIR + ROI[i] +"_SRG_4D.nii.gz")

    print "Program end..."
































