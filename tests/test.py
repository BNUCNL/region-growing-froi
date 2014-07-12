
from algorithm.regiongrowing import *
from algorithm.neighbor import *



if __name__ == "__main__":
    peak_coors = [[24, 36, 25],]
    seeds = Seeds(peak_coors)

    similarity_criteria = SimilarityCriteria('euclidean')
    stop_criteria = StopCriteria(10, 'size')

    image = nib.load("../data/S2/tstat1.nii.gz")
    affine = image.get_affine()
    image = image.get_data()

    conn = Connectivity(image.shape, 26)

    srg = SeededRegionGrowing(image, seeds, similarity_criteria, stop_criteria, conn)
    region = srg.grow()

    print region.label[:region.label_size,:]



"""
    nib.save(nib.Nifti1Image(output4d.astype(int), affine3d), "../tests/SeededRegionGrowing4D.nii.gz")

    print 'SeededRegionGrowing4D.nii.gz was created.'
"""

