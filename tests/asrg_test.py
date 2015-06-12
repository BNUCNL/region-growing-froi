
import time
import nibabel as nib

from algorithm.region_growing import *
from algorithm.neighbor import *
from algorithm.similarity_criteria import *
from algorithm.stop_criteria import *
from algorithm.region import *
from algorithm.optimizer import *

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

    srg = SeededRegionGrowing(similarity_criteria, stop_criteria)
    srg_region = srg.compute(region, image, threshold)

    result_image = np.zeros((image.shape[0], image.shape[1], image.shape[2], len(srg_region)))
    for i in range(len(srg_region)):
        labels = srg_region[i].get_label()
        result_image[labels[:, 0], labels[:, 1], labels[:, 2], i] = 1

    nib.save(nib.Nifti1Image(result_image, affine), "../data/S1/zstat1_srg.nii.gz")

    #AC
    optimizer_AC = Optimizer('AC')
    optimizer_AC_image = optimizer_AC.compute([srg_region], image)
    #PC
    # optimizer_PC = Optimizer('PC')
    # optimier_PC_image = optimizer_PC.compute(rsrg_region, image[..., j])

    index = optimizer_AC_image[0].argmax()
    optimal_regions = srg_region[index]
    labels = optimal_regions.get_label()

    result_image = np.zeros_like(image)
    result_image[labels[:, 0], labels[:, 1], labels[:, 2]] = 1
    nib.save(nib.Nifti1Image(result_image, affine), "../data/S1/zstat1_asrg_images.nii.gz")

    endtime = time.clock()
    print 'Cost time:: ', np.round((endtime - starttime), 3), ' s'
