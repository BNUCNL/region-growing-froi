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

    srg = SeededRegionGrowing(similarity_criteria, stop_criteria)
    srg_region = srg.compute(region, image, threshold)

    result_image = np.zeros((image.shape[0], image.shape[1], image.shape[2], len(srg_region)))
    for i in range(len(srg_region)):
        labels = srg_region[i].get_label()
        result_image[labels[:, 0], labels[:, 1], labels[:, 2], i] = 1

    nib.save(nib.Nifti1Image(result_image, affine), "../data/S1/zstat1_srg.nii.gz")

    endtime = time.clock()
    print 'Cost time:: ', np.round((endtime - starttime), 3), ' s'

regions = []
srg_region = srg.compute(region, image[..., j], threshold)
regions.append(srg_region)

#AC
optimizer_AC = Optimizer('AC')
optimizer_AC_image = optimizer_AC.compute(regions, image[..., j])
AC_roi_regions_result[j, :] = optimizer_AC_image[:]
title_AC = str(j + 1) + '. ' + lines[j ] + ' --  ' + ROI[i] + ' AC Analysis'
show_date_index_formatter(threshold, optimizer_AC_image[0, :], 'Threshold', 'AC Value', title_AC, 'g', True)

region_size_AC = threshold[optimizer_AC_image.reshape(len(threshold),).argmax()]



#PC
optimizer_PC = Optimizer('PC')
optimier_PC_image = optimizer_PC.compute(regions, image[..., j])
PC_roi_regions_result[j, :] = optimier_PC_image[:]

region_size_PC = threshold[optimier_PC_image.reshape(len(threshold),).argmax()]


#Save data to nii.gz
AC_labels = srg_region[optimizer_AC_image.reshape(len(threshold),).argmax()].get_label()
AC_region_image[AC_labels[:, 0], AC_labels[:, 1], AC_labels[:, 2], j] = 1
PC_labels = srg_region[optimier_PC_image.reshape(len(threshold),).argmax()].get_label()
PC_region_image[PC_labels[:, 0], PC_labels[:, 1], PC_labels[:, 2], j] = 1

