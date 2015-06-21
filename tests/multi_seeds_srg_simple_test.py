__author__ = 'zgf'

import time
import nibabel as nib

from algorithm.region_growing import *
from algorithm.similarity_criteria import *

if __name__ == "__main__":
    #load data
    starttime = time.clock()
    image = nib.load("../data/S1/zstat1.nii.gz")
    affine = image.get_affine()
    image = image.get_data()

    # subject_id		r_OFA			l_OFA			r_pFus			l_pFus
    # S1			(28, 29, 29)	(63, 26, 28)	(25, 41, 25)	(65, 42, 26)
    r_OFA_seed_coords = np.array([[28, 29, 29]])
    l_OFA_seed_coords = np.array([[63, 26, 28]])
    r_pFus_seed_coords = np.array([[25, 41, 25]])
    l_pFus_seed_coords = np.array([[65, 42, 26]])
    seed_coords = [r_OFA_seed_coords, l_OFA_seed_coords, r_pFus_seed_coords, l_pFus_seed_coords]

    ssl = {} #dict: key - neighbor_cord, value - neighbor_delta
    boundary = []

    #boundary -1 ,  unlabel 0, label 1...n
    result_image = np.zeros_like(image).astype(np.int)
    neighbor_element = SpatialNeighbor('connected', image.shape, 26)

    for i in range(len(seed_coords)):
        result_image[seed_coords[i][:, 0], seed_coords[i][:, 1], seed_coords[i][:, 2]] = i + 1
        neighbors = neighbor_element.compute(seed_coords[i])
        mean = image[result_image == i + 1].mean()
        for j in range(neighbors.shape[0]):
            neighbor_value = image[tuple(neighbors[j, :])]
            ssl[tuple(neighbors[j, :])] = abs(neighbor_value - mean)

    all_regions_size = 2000
    # while len(ssl_cords) > 0 or (result_image > 0).sum() < all_regions_size:
    while (result_image > 0).sum() < all_regions_size:
        min_delta_key = ssl.keys()[np.array(ssl.values()).argmin()]
        nearest_neighbor_cord = np.array(min_delta_key)
        print ' nearest_neighbor_cord: ', nearest_neighbor_cord

        neighbors = neighbor_element.compute(nearest_neighbor_cord)
        neighbor_values = result_image[neighbors[:, 0], neighbors[:, 1], neighbors[:, 2]]

        unique_values = np.unique(neighbor_values)
        if len(unique_values) != 2:
            boundary.append(nearest_neighbor_cord)
            result_image[tuple(nearest_neighbor_cord)] = -1 #boundary label value
        else:
            new_label = np.unique(neighbor_values)[1] #[0, new label]
            result_image[tuple(nearest_neighbor_cord)] = new_label
            #update the corresponding region mean
            new_region_mean = image[result_image == new_label].mean()

            for i in range(neighbors.shape[0]):
                cord = neighbors[i, :]
                value = result_image[tuple(cord)]

                if value == 0 and not ssl.has_key(tuple(cord)) :
                    ssl[tuple(cord)] = abs(image[tuple(cord)] - new_region_mean)
        del ssl[min_delta_key] #delete neighbor from ssl

        print '-------', (result_image > 0).sum(), '-----', len(boundary), '----', len(ssl), '--------'

    nib.save(nib.Nifti1Image(result_image, affine), "../data/S1/zstat1_multi_seeds_srg.nii.gz")

    endtime = time.clock()
    print 'Cost time:: ', np.round((endtime - starttime), 3), ' s'























