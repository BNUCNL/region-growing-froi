__author__ = 'zgf'

import nibabel as nib
import numpy as np
import multiprocessing

from skimage.segmentation import slic
from scipy.ndimage.morphology import binary_dilation
from configs import *

SUPERVOXEL_SEGMENTATION = 100000
SUBJECT_NUM = 70

images = nib.load(ACTIVATION_DATA_DIR)
affine = images.get_affine()
all_volumes = images.get_data()

left_barin_mask = nib.load(PROB_ROI_202_SUB_FILE + PROB_LEFT_BRAIN_FILE).get_data() > 0
right_barin_mask = nib.load(PROB_ROI_202_SUB_FILE + PROB_RIGHT_BRAIN_FILE).get_data() > 0

roi_peak_points = np.load(PEAK_POINTS_DIR + RESULT_NPY_FILE)

def compute_supervoxel(volume):
    gray_image = (volume - volume.min()) * 255 / (volume.max() - volume.min())

    slic_image = slic(gray_image.astype(np.float),
                      n_segments=SUPERVOXEL_SEGMENTATION,
                      slic_zero=True,
                      sigma=2,
                      multichannel =False,
                      enforce_connectivity=True)
    # nib.save(nib.Nifti1Image(slic_image, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + 'supervoxel.nii.gz')

    unique_values = np.unique(slic_image)
    size = np.zeros((len(unique_values), ))
    for i in range(len(unique_values)):
        size[i] = (slic_image == unique_values[i]).sum()

    print 'size.max(): ', size.max(), '  size.min(): ', size.min(), '  size.mean(): ', size.mean(),
    '  size.std(): ', size.std()

    return slic_image

def compute_slic_max_region_mean(volume, region_volume, slic_image):
    neighbor_slic = binary_dilation(region_volume)
    neighbor_slic[region_volume > 0] = 0

    neighbor_values = np.unique(slic_image[neighbor_slic > 0])
    # print 'neighbor_values: ', neighbor_values

    region_means = np.zeros((len(neighbor_values - 1), ))
    for i in range(len(neighbor_values)):
        if neighbor_values[i] !=0 :
            neighbor_slic[slic_image == neighbor_values[i]] = 1
            region_means[i] = volume[slic_image == neighbor_values[i]].mean()
        # print 'region_means: ',region_means
    # print 'neighbor_values[region_means.argmax(): ', neighbor_values[region_means.argmax()]

    return neighbor_slic, slic_image == neighbor_values[region_means.argmax()]

def supervoxel_based_regiongrowing(slic_image, volume, seed, size=10):
    seed = np.array(seed)
    seed_region = np.zeros_like(slic_image)
    seed_region[slic_image == slic_image[seed[0], seed[1], seed[2]]] = 1

    seed_regions = np.zeros((seed_region.shape[0], seed_region.shape[1], seed_region.shape[2], size))
    seed_regions[..., 0] = seed_region
    neighbor_slics = np.zeros((seed_region.shape[0], seed_region.shape[1], seed_region.shape[2], size))

    for i in range(0, size-1):
        neighbor_slic, best_parcel = compute_slic_max_region_mean(volume, seed_region, slic_image)
        seed_region[best_parcel] = 1
        seed_regions[..., i + 1] = seed_region
        neighbor_slics[..., i] = neighbor_slic

    neighbor_slics[..., size - 1] = compute_slic_max_region_mean(volume, seed_region, slic_image)[0]

    # nib.save(nib.Nifti1Image(seed_regions, affine),
    #          RW_AGGRAGATOR_RESULT_DATA_DIR + str(SUPERVOXEL_SEGMENTATION) + '_slic_regiongrowing.nii.gz')
    # nib.save(nib.Nifti1Image(neighbor_slics, affine),
    #          RW_AGGRAGATOR_RESULT_DATA_DIR + str(SUPERVOXEL_SEGMENTATION) + '_neighbor_slic_regiongrowing.nii.gz')

    return neighbor_slics, seed_regions

def compute_optional_region_based_AC_value(volume, regions, neighbor_slics):
    AC_values = np.zeros((regions.shape[3], ))
    for i in range(regions.shape[3]):
        AC_values[i] = volume[regions[..., i] > 0].mean() - volume[neighbor_slics[..., i] > 0].mean()

    # print 'AC_values: ', AC_values
    return regions[..., AC_values.argmax()]

def single_process(subject_index):
    result_volume = np.zeros((all_volumes.shape[0], all_volumes.shape[1], all_volumes.shape[2]))
    slic_image = compute_supervoxel(all_volumes[..., subject_index])

    for i in range(len(ROI)):
        seed = np.array([roi_peak_points[subject_index, i, :]]).astype(np.int)[0]
        neighbor_slics, regions = supervoxel_based_regiongrowing(slic_image, all_volumes[..., subject_index], seed, size=10)
        optimal_region = compute_optional_region_based_AC_value(all_volumes[..., subject_index], regions, neighbor_slics)
        result_volume[optimal_region > 0] = i + 1
    print 'subject_index: ', subject_index
    return result_volume, slic_image

if __name__ == "__main__":
    import datetime
    starttime = datetime.datetime.now()

    result_volumes = np.zeros((all_volumes.shape[0], all_volumes.shape[1], all_volumes.shape[2], SUBJECT_NUM))
    slic_images = np.zeros((all_volumes.shape[0], all_volumes.shape[1], all_volumes.shape[2], SUBJECT_NUM))

    #single process
    # for j in range(0, all_volumes.shape[3]):
    # # for j in range(0, 2):
    #     for i in range(len(ROI)):
    #         seed = np.array([roi_peak_points[j, i, :]]).astype(np.int)[0]
    #
    #         neighbor_slics, regions = supervoxel_based_regiongrowing(all_volumes[..., j], seed, size=10)
    #         optimal_region = compute_optional_region_based_AC_value(all_volumes[..., j], regions, neighbor_slics)
    #
    #         result_volumes[optimal_region > 0, j] = i + 1
    #
    #         print 'j: ', j, ' i: ', i, ' ROI: ', ROI[i]

    #multi process
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    pool_outputs = pool.map(single_process, range(0, SUBJECT_NUM))
    pool.close()
    pool.join()

    for subject_index in range(SUBJECT_NUM):
        result_volumes[..., subject_index], slic_images[..., subject_index] = pool_outputs[subject_index]

    slic_images[left_barin_mask, :] = 0
    slic_images[right_barin_mask, :] = 0

    nib.save(nib.Nifti1Image(result_volumes, affine),
             SSRG_RESULT_DOC_DATA_DIR + str(SUPERVOXEL_SEGMENTATION) + '_result_regions.nii.gz')
    nib.save(nib.Nifti1Image(slic_images, affine),
             SSRG_RESULT_DOC_DATA_DIR + str(SUPERVOXEL_SEGMENTATION) + '_slic_images.nii.gz')

    endtime = datetime.datetime.now()
    print 'time: ', (endtime - starttime)


















