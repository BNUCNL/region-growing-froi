import numpy as np
import nibabel as nib
from ..algorithm.regiongrowing import SeededRegionGrowing
from ..algorithm.regiongrowing import AverageContrast
from ..algorithm.regiongrowing import AdaptiveSRG

if __name__ == "__main__":
    t_image = nib.load('../data/S2/tstat1.nii.gz')
    data = t_image.get_data()
    A = AverageContrast(data, (24, 37, 25), 1000)
    new_image = A.grow()
    t_image._data = new_image
    nib.save(t_image, 'ACB_S2_image.nii.gz')
    print 'average contrast growing has been saved.'

    #B = PeripheralContrast(data, (26, 38, 25), 1000)
    #new_image = B.grow()
    #t_image._data = new_image
    #nib.save(t_image, 'PCB_S2_image')
    #print 'peripheral contrast growing has been saved.'

    #C = SeededRegionGrowing(data, (26, 38, 25), 200)
    #new_image = C.grow()
    #t_image._data = new_image
    #nib.save(t_image, 'SRG_S2_image.nii.gz')
    #print 'fixed threshold growing has been saved.'

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