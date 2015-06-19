__author__ = 'zgf'

import time
import nibabel as nib

from algorithm.region_growing import *
from algorithm.neighbor import *
from algorithm.similarity_criteria import *
from algorithm.stop_criteria import *
from algorithm.region import *

if __name__ == "__main__":
    #load data
    starttime = time.clock()
    image = nib.load("../data/S1/zstat1.nii.gz")
    affine = image.get_affine()
    image = image.get_data()

    seed_coords = np.array([[28, 29, 29]]) # r_OFA seed point: [28, 29, 29]
    neighbor_element = SpatialNeighbor('connected', image.shape, 26)
    region = Region(seed_coords, neighbor_element)

    similarity_criteria = SimilarityCriteria('euclidean')
    stop_criteria = StopCriteria('size')
    threshold = np.array((50, 100, 150, 200))

    slic_srg = SlicSRG(similarity_criteria, stop_criteria)

    result_volume = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
    slic_image = slic_srg.convert_image_to_supervoxel(image)
    nib.save(nib.Nifti1Image(slic_image, affine), "../data/S1/zstat1_slic_srg_supervoxel_image.nii.gz")

    neighbor_slics, regions = slic_srg.supervoxel_based_regiongrowing(slic_image, image, seed_coords)
    optimal_region_image = slic_srg.compute_optional_region_based_AC_value(image, regions, neighbor_slics)
    nib.save(nib.Nifti1Image(optimal_region_image, affine), "../data/S1/zstat1_slic_srg_optimal_image.nii.gz")

    endtime = time.clock()
    print 'Cost time:: ', np.round((endtime - starttime), 3), ' s'















