import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

DATA_NUM = 100
FFA_LABEL = 3

DATA_PATH = "/nfs/t2/fmricenter/volume/"
LABEL_PATH = "/nfs/t2/atlas/database/"
LABEL_POSTFIX = "/face-object/"
SUBJECT_POSTFIX = "/obj.gfeat/cope1.feat/stats/tstat1.nii.gz"
SUBJECT_LABELS = "face_z2.3.nii.gz"
SUBJECT_NUMBER = [ ]

output=['S0001(25,41,25)','S0004(25,36,29)','S0005(26,31,29)','S0006(23,38,23)','S0007(66,38,29)',\
        'S0009(25,39,24)','S0010(25,38,25)','S0012(25,41,26)','S0014(24,35,22)','S0016(22,35,24)',\
        'S0020(27,29,28)','S0025(68,36,24)','S0027(27,30,26)','S0034(23,37,24)','S0035(26,38,25)',\
        'S0036(24,38,27)','S0037(22,31,26)','S0038(23,34,26)','S0040(24,37,24)','S0041(24,37,25)',\
        'S0051(22,38,27)','S0053(25,36,29)','S0057(25,40,24)','S0059(24,38,24)','S0068(21,38,24)',\
        'S0069(24,29,28)','S0070(26,40,26)','S0074(24,39,26)','S0076(23,38,26)','S0077(21,40,22)',\
        'S0081(23,38,26)','S0085(23,39,23)','S0087(25,35,25)','S0091(23,38,30)','S0092(65,38,24)']

number=[116,64,21,13,166,239,172,48,148,783,54,27,94,917,138,135,25,187,\
       127,113,280,38,129,245,68,122,19,102,84,96,340,543,55,231,62]

def load_data():
    subject_images = [ ]
    labels_images = [ ]

    index = 0
    while True:
        index = index + 1
        if len(subject_images) >= DATA_NUM:
            break
        else:
            subject_num = "S" + '{:0>4}'.format(index)
            filepath = DATA_PATH + subject_num
            if os.path.exists(filepath):
                SUBJECT_NUMBER.append(subject_num)
                subject_images.append(nib.load(filepath + SUBJECT_POSTFIX).get_data())
                for item in os.listdir(LABEL_PATH + subject_num + LABEL_POSTFIX):
                    if "_ff" in item:
                        labels_images.append(nib.load(LABEL_PATH + subject_num + LABEL_POSTFIX + item).get_data())
                        break

    subject_images = np.array(subject_images)
    labels_images = np.array(labels_images)

    return subject_images, labels_images

def choose_seeds(subject_data, labels_images):
    seeds = [ ]
    valid_count = 0;

    for index in range(0, DATA_NUM):
        temp = np.zeros((subject_data.shape[1], subject_data.shape[2], subject_data.shape[3]))
        temp[labels_images[index, :, :, :] == FFA_LABEL] = subject_data[index, labels_images[index, :, :, :] == FFA_LABEL]
        value = temp[labels_images[index, :, :, :] == FFA_LABEL]

        if value.size == 0:
           seeds.append((0, 0, 0))
           valid_count = valid_count + 1
        else:
           seeds.append(np.unravel_index(temp.argmax(), subject_data[index, :, :, :].shape))

    result = np.array(seeds)
    result.shape
    np.savetxt('seeds.txt', result, delimiter=',',  fmt='%.0f')
    print 'Valid label: ', valid_count

    return seeds

def dice(volume1, volume2):
    """
    Computes the Dice coefficient, a measure of set similarity.

    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.

    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    """
    if volume1.shape != volume2.shape:
        raise ValueError("Shape mismatch: volume1 and volume2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(volume1, volume2)

    return 2. * intersection.sum() / (volume1.sum() + volume2.sum())

def region_growing(image, coordinate, output, NUM):
    """
    Give a coordinate ,return a region.
    """
    x, y, z = coordinate

    if not isinstance(image, np.ndarray):
        raise ValueError("The input image not belong to numpy ndarray.")

    image_shape = image.shape
    if(len(image_shape) != 3):
        raise ValueError("Valid data, not three demension.")

    inside = (x >= 0) and (x < image_shape[0]) and (y >= 0) and \
             (y < image_shape[1]) and (z >= 0) and (z < image_shape[2])
    if not inside:
        raise ValueError("The coordinate is out of the image range.")

    if NUM <= 0:
        raise ValueError("Valid Nnumber.")

    filename = output
    contents = []

    tmp_image = np.zeros_like(image)
    inner_image = np.zeros_like(image)
    inner_list = []
    contrast = []
    
    region_size = 1
    origin_t = image[x, y, z]
    inner_list = inner_list + [origin_t]

    neighbor_free = 10000
    neighbor_pos = -1
    neighbor_list = np.zeros((neighbor_free, 4))

    offsets = [[1, 0, 0],
               [-1, 0, 0],
               [0, 1, 0],
               [0, -1, 0],
               [0, 0, -1],
               [0, 0, 1],
               [1, 1, 0],
               [1, 1, 1],
               [1, 1, -1],
               [0, 1, 1],
               [-1, 1, 1],
               [1, 0, 1],
               [1, -1, 1],
               [-1, -1, 0],
               [-1, -1, -1],
               [-1, -1, 1],
               [0, -1, -1],
               [1, -1, -1],
               [-1, 0, -1],
               [-1, 1, -1],
               [0, 1, -1],
               [0, -1, 1],
               [1, 0, -1],
               [1, -1, 0],
               [-1, 0, 1],
               [-1, 1, 0]]


    while region_size <= NUM:
        for i in range(26):
            xn = x + offsets[i][0]
            yn = y + offsets[i][1]
            zn = z + offsets[i][2]

            inside = (xn >= 0) and (xn < image_shape[0]) and (yn >= 0) and (yn < image_shape[1]) \
                     and (zn >= 0) and (zn < image_shape[2])
            
            if inside and tmp_image[xn, yn, zn]==0:
                neighbor_pos = neighbor_pos + 1
                neighbor_list[neighbor_pos] = [xn, yn,zn,image[xn, yn, zn]]
                tmp_image[xn, yn, zn] = 1

        out_boundary = neighbor_list[np.nonzero(neighbor_list[:, 3]), 3]
        contrast = contrast + [np.mean(np.array(inner_list)) - np.mean(out_boundary)]


        entry = '%.6f' % (np.mean(np.array(inner_list)) - np.mean(out_boundary))
        contents.append(entry)
        fobj = open(filename,'w')
        fobj.writelines(['%s%s' % (eachline, os.linesep) for eachline in contents])


        tmp_image[x, y, z] = 2
        inner_image[x, y, z] = image[x, y, z]
        region_size += 1

        if (neighbor_pos + 100 > neighbor_free):
            neighbor_free += 10000
            new_list = np.zeros((10000, 4))
            neighbor_list = np.vstack((neighbor_list, new_list))
            #if the longth of neighbor_list is not enough,add another 10000       
            
        distance = np.abs(neighbor_list[:neighbor_pos+1, 3] - np.tile(origin_t, neighbor_pos + 1))
        index = distance.argmin()            
        x,y,z = neighbor_list[index][:3]
        inner_list = inner_list + [image[x, y, z]]
       
        neighbor_list[index] = neighbor_list[neighbor_pos]
        neighbor_pos -= 1
   
    return inner_image, region_size-1, contrast


def average_contrast():
    for i in range(0,35):
        t_img = nib.load(image_index[i])
        t_data = t_img.get_data()
        region, num, contrast = region_growing(t_data, seeds[i], output[i], number[i])
        # region,num,contrast = region_growing(t_data,seeds[i],output[i])
        t_img._data = region
        y1 = contrast
        nib.save(t_img, output[i] + ".nii.gz")
        print num

        x1 = range(1, num + 1)
        plt.plot(x1, y1, label='average contrast')

        plt.show()

if   __name__  ==  "__main__":
    subject_data, labels_images = load_data()
    print choose_seeds(subject_data, labels_images)
   
