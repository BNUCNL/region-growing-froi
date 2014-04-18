 # emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
 # vi: set ft=python sts=4 ts=4 sw=4 et:
import os
import numpy as np


def inside(coors,image_shape):
    """
    Give a coordinate ,return True or False.
    """
    return  (coors[0] >= 0) and (coors[0] < image_shape[0]) and \
            (coors[1] >= 0) and (coors[1] < image_shape[1]) and \
            (coors[2] >= 0) and (coors[2] < image_shape[2])


def fixed(region_var,single_var,beta_list,region_size):
    """
    Fixed effect.
    """
    region_var = region_var + single_var
    return region_var


def random(region_var,single_var,beta_list,region_size):
    """
    Random effect.
    """
    if region_size<10:
        region_var=region_var + single_var
    else:
        region_var = region_size * np.var(beta_list)
    return region_var


def mixed(region_var,single_var,beta_list,region_size):
    """
    Mixed effect.
    """    
    region_var = (region_var+single_var) + region_size * np.var(beta_list)  
    return region_var


offsets = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],\
           [0,0,-1],[0,0,1],[1,1,0],[1,1,1],\
           [1,1,-1],[0,1,1],[-1,1,1],[1,0,1],\
           [1,-1,1],[-1,-1,0],[-1,-1,-1],[-1,-1,1],\
           [0,-1,-1],[1,-1,-1],[-1,0,-1],[-1,1,-1],\
           [0,1,-1],[0,-1,1],[1,0,-1],[1,-1,0],\
           [-1,0,1],[-1,1,0]]
      

def region_growing(seed,beta_image,var_image,nbsize,mode,output):
    """
    Give a coordinate ,return a region.
    """
    image_shape = beta_image.shape
    if image_shape!=var_image.shape:
        print "The belta_image and var_image can't match,Please check it."
        return False

    if inside(seed,image_shape)!=True:
        print "The coordinate is out of the image range."
        return False

    if mode == 'fixed':
        mode=fixed
    elif mode == 'mixed':
        mode=mixed
    elif mode == 'random':
        mode=random
    else:
        print 'mode must be fixed/random/mixed'
        return False

    x,y,z = seed
    seed_t = beta_image[x,y,z]/np.sqrt(var_image[x,y,z])
    voxel_t = [seed_t]
    #print seed_t
    flag_image = np.zeros_like(beta_image)
    flag_image[seed] = 2
    
    t_image = np.zeros_like(beta_image)
    t_image[x,y,z] = seed_t
    beta_list = [beta_image[x,y,z]]

    region_size = 1
    region_beta = beta_image[x,y,z]
    region_var = var_image[x,y,z]
    region_t_list = [seed_t]
    region_t_change=0
    region_t_change_list=[]
    

    filename = output
    contents = []
    entry = '%d %d %d   %.6f   %.6f   %.6f   %.6f' % (x,y,z,beta_image[x,y,z],var_image[x,y,z],seed_t,region_t_change)
    contents.append(entry)
    fobj = open(filename,'w')
    fobj.writelines(['%s%s' % (eachline, os.linesep) for eachline in contents])

    #allocate space for neighbor_list
    neighbor_free = 10000
    neighbor_pos = -1
    neighbor_list = np.zeros((neighbor_free,4))

    while region_size < 10:
      
        for i in range(nbsize):
            set0,set1,set2 = offsets[i]
            xn,yn,zn = x+set0,y+set1,z+set2           
            if inside((xn,yn,zn),image_shape) and var_image[xn,yn,zn]!=0 and flag_image[xn,yn,zn]==0:
                neighbor_pos = neighbor_pos+1
                tn = beta_image[xn,yn,zn]/np.sqrt(var_image[xn,yn,zn],dtype=np.float64)
                neighbor_list[neighbor_pos] = [xn,yn,zn,tn]
                flag_image[xn,yn,zn] = 1
            
        distance = np.abs(neighbor_list[:neighbor_pos+1,3] - np.tile(seed_t,neighbor_pos+1))
        #distance = np.abs(neighbor_list[:neighbor_pos+1,3] - np.tile(np.mean(voxel_t),neighbor_pos+1))
        index = distance.argmin()

        x,y,z = neighbor_list[index][:3]
        flag_image[x,y,z] = 2
        t_image[x,y,z]=beta_image[x,y,z]/np.sqrt(var_image[x,y,z],dtype=np.float64)
        beta_list = beta_list + [beta_image[x,y,z]]
        voxel_t = voxel_t + [t_image[x,y,z]] 
          
        region_size += 1
        last_region_beta = region_beta
        last_region_var = region_var
        region_beta = region_beta + beta_image[x,y,z]


        region_var = mode(region_var,var_image[x,y,z],beta_list,region_size)

        region_t = region_beta / np.sqrt(region_var,dtype=np.float64)
        #print region_t
        region_t_change = region_t - last_region_beta/np.sqrt(last_region_var,dtype=np.float64)
        region_t_list = region_t_list + [region_t]
        region_t_change_list = region_t_change_list + [region_t_change]        

        entry = '%d %d %d   %.6f   %.6f   %.6f   %.6f' % (x,y,z,beta_image[x,y,z],var_image[x,y,z],region_t,region_t_change)
        contents.append(entry)
        fobj = open(filename,'w')
        fobj.writelines(['%s%s' % (eachline, os.linesep) for eachline in contents])
       
        neighbor_list[index] = neighbor_list[neighbor_pos]
        neighbor_pos -= 1

    fobj.close()
    return t_image,region_size,region_t_list,region_t_change_list

