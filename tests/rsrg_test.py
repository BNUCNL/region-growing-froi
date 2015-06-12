import time

import nibabel as nib

from algorithm.region_growing import *
from algorithm.neighbor import *
from algorithm.similarity_criteria import *
from algorithm.stop_criteria import *
from algorithm.region import *
from algorithm.aggregator import *
from algorithm.optimizer import *

if __name__ == "__main__":
    #load data
    starttime = time.clock()

    image = nib.load("../data/S1/zstat1.nii.gz")
    affine = image.get_affine()
    image = image.get_data()

    mask = nib.load("../data/prior/prob_rFFA.nii.gz")
    mask = mask.get_data()

    seed_coords = np.array(np.nonzero(mask >= 0.5)).T
    print 'seed_coords.shape => ', seed_coords.shape
    neighbor_element = SpatialNeighbor('connected', image.shape, 26)
    region = Region(seed_coords, neighbor_element)

    similarity_criteria = SimilarityCriteria('euclidean')
    stop_criteria = StopCriteria('size')
    threshold = np.array((200, 300, 400, 500))

    # similarity_criteria.set_rand_neighbor_prop(0.7)
    seed_sampling_num = 10
    rsrg = RandomSRG(similarity_criteria, stop_criteria, seed_sampling_num)
    rsrg_region = rsrg.compute(region, image, threshold)

    #AC
    optimizer_AC = Optimizer('AC')
    optimizer_AC_image = optimizer_AC.compute(rsrg_region, image)

    optimal_regions = []
    optimal_images = np.zeros((image.shape[0], image.shape[1], image.shape[2], len(optimizer_AC_image)))
    for i in range(len(optimizer_AC_image)):
        index = optimizer_AC_image[i].argmax()
        optimal_regions.append([rsrg_region[i][index]])
        labels = rsrg_region[i][index].get_label()
        print 'len(labels): => ', len(labels)
        optimal_images[labels[:, 0], labels[:, 1], labels[:, 2], i] = 1
        print 'i: ', i, '   => ', (optimal_images[..., i] == 1).sum(), '    index => ', index
    nib.save(nib.Nifti1Image(optimal_images, affine), "../data/S1/zstat1_rsrg_optimal_images.nii.gz")

    #PC
    # optimizer_PC = Optimizer('PC')
    # optimier_PC_image = optimizer_PC.compute(rsrg_region, image[..., j])

    aggregator = Aggregator('MWA')
    agg_image = aggregator.compute(optimal_regions, image) #Only generate one probility region
    result_image = agg_image[..., 0]
    nib.save(nib.Nifti1Image(result_image, affine), "../data/S1/zstat1_rsrg_optimal_aggragator.nii.gz")

    endtime = time.clock()
    print 'Cost time:: ', np.round((endtime - starttime), 3), ' s'


































