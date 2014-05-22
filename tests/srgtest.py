__author__ = 'zgf'
import os
import numpy as np
import nibabel as nib
from algorithm.regiongrowing import *

if __name__ == "__main__":
    print os.getcwd()

    img4d = nib.load("../data/4DVolume.nii.gz")
    affine4d = img4d.get_affine()
    target_image_4d = img4d.get_data()

    img3d = nib.load("../data/S2/tstat1.nii.gz")
    affine3d = img3d.get_affine()
    target_image_3d = img3d.get_data()

    img2d = nib.load("../data/2DSlice.nii.gz")
    affine2d = img2d.get_affine()
    target_image_2d = img2d.get_data()

    cords3d = np.array([24, 36, 25])
    seeds3d = Seeds(cords3d)
    similarity_criteria = NeighborSimilarity(metric='euclidean',)
    stop_criteria = StopCriteria(name='region_size', threshold=300)
    connectivity = Connectivity('6')

    srg = SeededRegionGrowing(target_image_4d, seeds3d, similarity_criteria, stop_criteria, connectivity)
    output4d = srg.grow().get_current_region()
    nib.save(nib.Nifti1Image(output4d.astype(int), affine3d), "../tests/SeededRegionGrowing4D.nii.gz")
    print 'SeededRegionGrowing4D.nii.gz was created.'

    srg.set_target_image(target_image_3d)
    output3d = srg.grow().get_current_region()
    nib.save(nib.Nifti1Image(output3d.astype(int), affine3d), "../tests/SeededRegionGrowing3D.nii.gz")
    print 'SeededRegionGrowing3D.nii.gz was created.'

    srg.get_connectivity().set_name('8')
    seeds2d = Seeds(np.array([24, 36]))
    srg.set_seeds(seeds2d)
    srg.set_target_image(target_image_2d)
    output2d = srg.grow().get_current_region()
    nib.save(nib.Nifti1Image(output2d.astype(int), affine2d), "../tests/SeededRegionGrowing2D.nii.gz")

    print 'SeededRegionGrowing2D.nii.gz was created.'
    print '------------------END------------------'