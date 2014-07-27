import nibabel as nib

from algorithm.regiongrowing import *
from algorithm.neighbor import *


if __name__ == "__main__":
    mask = nib.load("../data/prob_rFFA.nii.gz")
    mask = mask.get_data()
    seed_coords = np.array(np.nonzero(mask >= 0.6))
    seeds = Seeds(seed_coords.T)

    similarity_criteria = SimilarityCriteria('euclidean')
    stop_criteria = StopCriteria(300, 'size')

    image = nib.load("../data/S2/tstat1.nii.gz")
    affine = image.get_affine()
    image = image.get_data()

    spatial_neighbor = SpatialNeighbor('connected', image.shape, 26)

    srg = RandomSRG(image, seeds, similarity_criteria, stop_criteria, spatial_neighbor)
    region = srg.grow()

    #print region.shape

    aggregator = Aggregator('MWA')
    srg_image = aggregator.compute(region, image)


    #srg_image = np.zeros_like(image, dtype=int)
    #srg_image[region_label[:, 0], region_label[:, 1], region_label[:, 2]] = 1
    #nib.save(nib.Nifti1Image(srg_image, affine), "../data/S2/RSRG3d.nii.gz")

