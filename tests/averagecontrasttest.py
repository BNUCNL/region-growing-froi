import numpy as np
import nibabel as nib
from ..algorithm.regiongrowing import AverageContrast

if __name__ == "__main__":
    t_image = nib.load('../data/S2/tstat1.nii.gz')
    data = t_image.get_data()
    A = AverageContrast(data, (26,38,25), 1000)
    new_image = A.grow()
    t_image._data = new_image
    nib.save(t_image,'ACB_S2_image.nii.gz')
    print 'average contrast growing has been saved.'

    #B = Peripheral_contrast(data, (26,38,25), 1000)
    #new_image = B._grow(data, (26,38,25), 1000)
    #t_image._data = new_image
    #nib.save(t_image,'PCB_S2_image')
    #print 'peripheral contrast growing has been saved.'

    #C = fixed_region_grow(data,(26,38,25),200)
    #new_image = C._grow(data,(26,38,25),200)
    #t_image._data = new_image
    #nib.save(t_image,'fixed_thres_S2_image.nii.gz')
    #print 'fixed threshold growing has been saved.'
