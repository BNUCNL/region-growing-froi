import nibabel as nib

from algorithm.regiongrowing import *
from algorithm.neighbor import *


if __name__ == "__main__":
    mask = nib.load("../data/prob_rFFA.nii.gz")
    mask = mask.get_data()

    seed_coords = np.array(np.nonzero(mask >= 0.6))
    neighbor_element = SpatialNeighbor('connected', mask.shape, 26)
    region = Region(seed_coords.T, neighbor_element)

    similarity_criteria = SimilarityCriteria('euclidean', 0.8)
    stop_criteria = StopCriteria('size')

    image = nib.load("../data/S2/tstat1.nii.gz")
    affine = image.get_affine()
    image = image.get_data()

    #srg = SeededRegionGrowing(similarity_criteria, stop_criteria)
    #threshold = np.array((10,50))
    #region = srg.compute(region, image, threshold)

    similarity_criteria.set_rand_neighbor_prop(0.7)
    rsrg = RandomSRG(similarity_criteria, stop_criteria, 10)
    threshold = np.array((10, 50))
    rsrg.compute(region, image, threshold)






    #print region.shape

    #aggregator = Aggregator('MWA')
    #srg_image = aggregator.compute(region, image)


    #srg_image = np.zeros_like(image, dtype=int)
    #srg_image[region_label[:, 0], region_label[:, 1], region_label[:, 2]] = 1
    #nib.save(nib.Nifti1Image(srg_image, affine), "../data/S2/RSRG3d.nii.gz")

