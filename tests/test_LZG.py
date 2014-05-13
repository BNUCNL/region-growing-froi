import numpy as np
import nibabel as nib
from ..algorithm.regiongrowing import RegionOptimizer
from ..algorithm.regiongrowing import AdaptiveSRG

if __name__ == "__main__":
    t_image = nib.load('../data/S2/tstat1.nii.gz')
    header = t_image.get_affine()
    data = t_image.get_data()

    A = RegionOptimizer(data, (24, 37, 25), 300, 26)
    region = A.optimal_average_contrast()
    nib.save(nib.Nifti1Image(region, header), 'Max_ACB_S2_image.nii.gz')



if __name__ == "__main__":
    t_image = nib.load('../data/S2/tstat1.nii.gz')
    data = t_image.get_data()
    A = AdaptiveSRG(data, (24, 37, 25), 1000, 26)
    regionsequence = A.grow()

    region = A.region_optimizer('average')
    t_image._data = region
    nib.save(t_image, 'Max_ACB_S2_image.nii.gz')

    for i in range(20):
        t_image._data = regionsequence[i]
        j = str(i)
        nib.save(t_image, j + 'ACB_S2_image.nii.gz')
        print 'average contrast growing has been saved.'