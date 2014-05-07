
import os
import sys
from math import sqrt
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

N=1000
image_index = ["data/S1/tstat1.nii.gz","data/S2/tstat1.nii.gz",\
               "data/S3/tstat1.nii.gz","data/S4/tstat1.nii.gz",\
               "data/S5/tstat1.nii.gz"]
seeds=[(26,38,25),(24,37,24),(24,37,25),(24,38,24),(23,38,26)]
output=['S1(26,38,25)','S2(24,37,24)','S3(24,37,25)','S4(24,38,24)','S5(23,38,26)']

offsets = [[1,0,0],[-1,0,0],\
          [0,1,0],[0,-1,0],\
          [0,0,-1],[0,0,1],\
          [1,1,0],[1,1,1],\
          [1,1,-1],[0,1,1],\
          [-1,1,1],[1,0,1],\
          [1,-1,1],[-1,-1,0],\
          [-1,-1,-1],[-1,-1,1],\
          [0,-1,-1],[1,-1,-1],\
          [-1,0,-1],[-1,1,-1],\
          [0,1,-1],[0,-1,1],\
          [1,0,-1],[1,-1,0],\
          [-1,0,1],[-1,1,0]]

def inside(coordinate,image_shape):
    """
    whether the coordinate is in the image,return True or False.
    """
    return  (coordinate[0] >= 0) and (coordinate[0] < image_shape[0]) and \
            (coordinate[1] >= 0) and (coordinate[1] < image_shape[1]) and \
            (coordinate[2] >= 0) and (coordinate[2] < image_shape[2])

def contrast_max(image,seed,N,output):
    """
    find the max average contrast point.
    display the contrast curve in grow process
    """
    x,y,z = seed
    image_shape = image.shape
    if inside(seed,image_shape)!=True:
        print "The seed is out of the image range."
        return False

    filename = output
    contents = []

    contrast = []
    region_size = 1
    origin_t = image[x,y,z]
    inner_list = [origin_t]
    tmp_image = np.zeros_like(image)

    neighbor_free = 10000
    neighbor_pos = -1
    neighbor_list = np.zeros((neighbor_free,4))

    while region_size <= N:
        for i in range(26):
            set0,set1,set2 = offsets[i]
            xn,yn,zn = x+set0,y+set1,z+set2  
            if inside((xn,yn,zn),image_shape) and tmp_image[xn,yn,zn]==0:
                neighbor_pos = neighbor_pos+1
                neighbor_list[neighbor_pos] = [xn,yn,zn,image[xn,yn,zn]]   
                tmp_image[xn,yn,zn] = 1

        out_boundary = neighbor_list[np.nonzero(neighbor_list[:,3]),3]
        contrast = contrast + [np.mean(np.array(inner_list)) - np.mean(out_boundary)]

        entry = '%.6f' % (np.mean(np.array(inner_list)) - np.mean(out_boundary))
        contents.append(entry)
        fobj = open(filename,'w')
        fobj.writelines(['%s%s' % (eachline, os.linesep) for eachline in contents])

        tmp_image[x,y,z] = 2       
        region_size += 1

        if (neighbor_pos+100 > neighbor_free):   
            neighbor_free +=10000                
            new_list = np.zeros((10000,4)) 
            neighbor_list = np.vstack((neighbor_list,new_list)) 
            #if the longth of neighbor_list is not enough,add another 10000    

        distance = np.abs(neighbor_list[:neighbor_pos+1,3] - np.tile(origin_t,neighbor_pos+1))        
        index = distance.argmin()        x,y,z = neighbor_list[index][:3]
        inner_list = inner_list + [image[x,y,z]]
        neighbor_list[index] = neighbor_list[neighbor_pos]
        neighbor_pos -= 1

    return contrast,np.array(contrast).argmax()+1

def region_growing(image,seed,N,output):
    """
    Give a coordinate ,return a region.
    """
    x,y,z = seed
    image_shape = image.shape

    if inside(seed,image_shape)!=True:
        print "The seed is out of the image range."
        return False
    
    region_size = 1
    origin_t = image[x,y,z]

    contrast,Num = contrast_max(image,seed,N,output)
    tmp_image = np.zeros_like(image)
    inner_image = np.zeros_like(image)

    neighbor_free = 10000
    neighbor_pos = -1
    neighbor_list = np.zeros((neighbor_free,4))

    while region_size <= Num:
        for i in range(26):
            xn = x + offsets[i][0]
            yn = y + offsets[i][1]
            zn = z + offsets[i][2]
            if inside((xn,yn,zn),image_shape) and tmp_image[xn,yn,zn]==0:
                neighbor_pos = neighbor_pos+1
                neighbor_list[neighbor_pos] = [xn,yn,zn,image[xn,yn,zn]]   
                tmp_image[xn,yn,zn] = 1

        tmp_image[x,y,z] = 2
        inner_image[x,y,z] = image[x,y,z]        
        region_size += 1
          
        distance = np.abs(neighbor_list[:neighbor_pos+1,3] - np.tile(origin_t,neighbor_pos+1))        
        index = distance.argmin()                    x,y,z = neighbor_list[index][:3]
        neighbor_list[index] = neighbor_list[neighbor_pos]
        neighbor_pos -= 1
   
    return inner_image,region_size-1,contrast

for i in range(0,5):
    t_img=nib.load(image_index[i])
    t_data=t_img.get_data() 
    region,num,contrast = region_growing(t_data,seeds[i],N,output[i])
    t_img._data = region
    y1 = contrast
    nib.save(t_img,output[i]+".nii.gz")
    print num
    plt.plot(np.array(contrast)[:])
    plt.xlabel('voxels')
    plt.ylabel('average contrast')
    plt.show()


