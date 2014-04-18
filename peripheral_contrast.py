
import os
import sys
from math import sqrt
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

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

def inner_boundary(flag_image,inner_region_cor):
    """
    find the inner boundary of the region.
    """
    inner_b = []
    for i in inner_region_cor:
        if is_neiflag(flag_image,i,1):
            if inner_b == []:
                inner_b = i
            else:
                inner_b= np.vstack((inner_b,i))
    #print inner_b
    return inner_b

def is_neiflag(flag_image,coordinate,flag):
    """
    if coordinate has a neighbor with certain flag return True,else False.
    """
    x,y,z = coordinate
    for j in range(26):
        xn = x + offsets[j][0]
        yn = y + offsets[j][1]
        zn = z + offsets[j][2]
        if flag_image[xn,yn,zn]==flag:
            return True
    return False


def region_growing(image,coordinate,out_put):
    """
    Give a coordinate ,return a region.
    """
    x,y,z = coordinate
    image_shape = image.shape
    inside = (x>=0)and(x<image_shape[0])and(y>=0)and\
             (y<image_shape[1])and(z>=0)and(z<image_shape[2])
    if inside!=True:
        print "The coordinate is out of the image range."
        return False

    filename = out_put
    contents = []

    tmp_image = np.zeros_like(image)
    sub_image = np.zeros_like(image)
    contrast = []
    
    region_size = 1
    origin_t = image[x,y,z]

    default_space = 10000
    outer_pos = -1
    inner_pos = -1
    outer_boundary_list = np.zeros((default_space,4))
    inner_list = np.zeros((default_space,4))

    while region_size <= 1000:

        inner_pos = inner_pos + 1
        inner_list[inner_pos] = [x,y,z,image[x,y,z]]

        for i in range(26):
            xn = x + offsets[i][0]
            yn = y + offsets[i][1]
            zn = z + offsets[i][2]
   
            inside = (xn>=0)and(xn<image_shape[0])and(yn>=0)and(yn<image_shape[1]) \
                     and(zn>=0)and(zn<image_shape[2])  
            
            if inside and tmp_image[xn,yn,zn]==0:
                outer_pos = outer_pos+1
                outer_boundary_list[outer_pos] = [xn,yn,zn,image[xn,yn,zn]]   
                tmp_image[xn,yn,zn] = 1

        outer_boundary = outer_boundary_list[np.nonzero(outer_boundary_list[:,3]),3]
        inner_region_cor = inner_list[np.nonzero(inner_list[:,3]),:3][0]
        inner_boundary_cor = inner_boundary(tmp_image,np.array(inner_region_cor))
        #print inner_boundary_cor

        inner_boundary_val = []
        if len(inner_boundary_cor.shape) == 1:
            inner_boundary_val = inner_boundary_val + [image[inner_boundary_cor[0], \
                                 inner_boundary_cor[1],inner_boundary_cor[2]]]
        else:
            for i in inner_boundary_cor:
                inner_boundary_val = inner_boundary_val + [image[i[0],i[1],i[2]]]

        contrast = contrast + [np.mean(inner_boundary_val) - np.mean(outer_boundary)]
        
        entry = '%.6f' % (np.mean(inner_boundary_val) - np.mean(outer_boundary))
        contents.append(entry)
        fobj = open(filename,'w')
        fobj.writelines(['%s%s' % (eachline, os.linesep) for eachline in contents])

        tmp_image[x,y,z] = 2
        sub_image[x,y,z] = image[x,y,z]
        region_size += 1

        if (outer_pos+100 > default_space):
            default_space +=10000
            new_list = np.zeros((10000,4))
            outer_boundary_list = np.vstack((outer_boundary_list,new_list))
                       
        distance = np.abs(outer_boundary_list[:outer_pos+1,3] - np.tile(origin_t,outer_pos+1))
        index = distance.argmin()        x,y,z = outer_boundary_list[index][:3]
       
        outer_boundary_list[index] = outer_boundary_list[outer_pos]
        outer_pos -= 1
   
    return sub_image,region_size-1,contrast


for i in range(0,35):
    t_img=nib.load(image_index[i])
    t_data=t_img.get_data() 
    region,num,contrast = region_growing(t_data,seeds[i],output[i])
    t_img._data = region

    y1 = contrast
    #nib.save(t_img,output+".nii.gz")
    print num
    x1 = range(1,num+1)
    plt.plot(x1,y1,label='peripheral contrast')

    plt.show()

