import nibabel as nib

from algorithm.regiongrowing import *
from algorithm.neighbor import *


if __name__ == "__main__":
    mask = nib.load("../data/S2/tstat1.nii.gz")
    mask = mask.get_data()
    [x, y, z] = np.nonzero(mask)
    seeds = Seeds([x, y, z])

    similarity_criteria = SimilarityCriteria('euclidean')
    stop_criteria = StopCriteria(300, 'size')

    image = nib.load("../data/S2/tstat1.nii.gz")
    affine = image.get_affine()
    image = image.get_data()

    spatial_neighbor = SpatialNeighbor('connected', image.shape, 26)
    aggregator = Aggregator('DA')

    srg = RandomSRG(image, seeds, similarity_criteria, stop_criteria, spatial_neighbor, aggregator)
    region = srg.grow()

    region_label = region.label
    print region_label

    srg_image = np.zeros_like(image, dtype=int)
    srg_image[region_label[:, 0], region_label[:, 1], region_label[:, 2]] = 1
    nib.save(nib.Nifti1Image(srg_image, affine), "../data/S2/SRG3d.nii.gz")

