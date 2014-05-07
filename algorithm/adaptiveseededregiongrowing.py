__author__ = 'zgf'
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

DATA_NUM = 30
FFA_LABEL = 3

DATA_PATH = "/nfs/t2/fmricenter/volume/"
PROBABILITY_PATH = "/nfs/t2/atlas/database/"
PROBABILITY_POSTFIX = "/face-object/"
SUBJECT_POSTFIX = "/obj.gfeat/cope1.feat/stats/tstat1.nii.gz"
SUBJECT_LABELS = "face_z2.3.nii.gz"
SUBJECT_NUMBER = []

output=['S0001(25,41,25)','S0004(25,36,29)','S0005(26,31,29)','S0006(23,38,23)','S0007(66,38,29)', \
        'S0009(25,39,24)','S0010(25,38,25)','S0012(25,41,26)','S0014(24,35,22)','S0016(22,35,24)', \
        'S0020(27,29,28)','S0025(68,36,24)','S0027(27,30,26)','S0034(23,37,24)','S0035(26,38,25)', \
        'S0036(24,38,27)','S0037(22,31,26)','S0038(23,34,26)','S0040(24,37,24)','S0041(24,37,25)', \
        'S0051(22,38,27)','S0053(25,36,29)','S0057(25,40,24)','S0059(24,38,24)','S0068(21,38,24)', \
        'S0069(24,29,28)','S0070(26,40,26)','S0074(24,39,26)','S0076(23,38,26)','S0077(21,40,22)', \
        'S0081(23,38,26)','S0085(23,39,23)','S0087(25,35,25)','S0091(23,38,30)','S0092(65,38,24)']


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
                print filepath

    subject_images = np.array(subject_images)
    labels_images = np.array(labels_images)

    return subject_images

def choose_seeds(subject_data, labels_images):
    seeds = [ ]
    valid_count = 0

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



if __name__ == "__main__":
    subject_data = load_data()
    # probability_images = nib.load('').get_data()
    # print choose_seeds(subject_data, probability_images)
    print '--------------------------END-------------------------'