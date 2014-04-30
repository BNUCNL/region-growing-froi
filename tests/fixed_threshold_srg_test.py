__author__ = 'zgf'
import os
import numpy as np
import nibabel as nib
from algorithm.region_growing import FixedThresholdSRG

SUBJECT_FILE_PATH = "../data/S1/tstat1.nii.gz"

if __name__ == "__main__":
    print os.getcwd()
    img = nib.load(SUBJECT_FILE_PATH)
    affine = img.get_affine()
    target_image = img.get_data()
    seed = [(25, 41, 25)]
    fixed_threshold_srg = FixedThresholdSRG(target_image, seed, stop_type='difference', value=5, connectivity='6')
    output = fixed_threshold_srg.grow()
    print '--------------------------END-------------------------'