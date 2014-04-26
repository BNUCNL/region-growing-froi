# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import neighbor as nb

def region_growing_froi(seed,beta_image,var_image,model_type,nbsize):
    """
    Give a seed coordinate ,return a region.
    model: fe,re,me
    """
    if not isinstance(seed,np.ndarray):
        seed = np.array(seed)


    image_shape = beta_image.shape 
    if var_image.shape != image_shape:
        raise ValueError("The beta_image and var_image are not matched.")
    elif not inside(seed,image_shape):
        raise ValueError("The seed is out of the image")
        
    if not isinstance(seed,np.ndarray):
        seed = np.array(seed)

    if model_type == 'fe':
        model = fe
    elif model_type == 're':
        model = re
    elif model_type == 'me':
        model = me
    else:
        raise ValueError("Model type for selectivity should be fe, re or me")

    # init the seed    
    sx,sy,sz = seed
    beta = beta_image[sx,sy,sz]
    var = var_image[sx,sy,sz]
    t  =  beta/np.sqrt(var,dtype=np.float64) 
    seed_value = t
    
    # init the region    
    region = np.array([beta,var,t])
    region = region.reshape(1,-1)
    voxel = region


    # create a empty image to store the t value for the region
    region_t_image = np.zeros_like(beta_image)
    region_t_image[sx,sy,sz] = t
     
    # init the neighbors
    offsets = nb.pixelconn(3,nbsize).compute_offsets()
    offsets = np.delete(offsets.T,(0),axis=0)
    neighbor = np.tile(seed,(nbsize,1)) + offsets
    neighbor = neighbor[inside(neighbor,image_shape),:]
    neighbor_t = beta_image[neighbor[:,0],neighbor[:,1],neighbor[:,2]]/ \
            np.sqrt(var_image[neighbor[:,0],neighbor[:,1],neighbor[:,2]])

    neighbor_list  = np.hstack((neighbor,neighbor_t.reshape(-1,1)))
     
    # create a empty image to flag the voxel
    flag_image = np.zeros(image_shape,int)  
    flag_image[neighbor[:,0],neighbor[:,1],neighbor[:,2]] = 1
    flag_image[sx,sy,sz] = 2

    t_change = 0

    while region.shape[0]  < 10:
        dist = np.abs(neighbor_list[:,3] - np.tile(seed_value,neighbor_list.shape[0]))
        #print dist
        index = dist.argmin()    
        new_voxel = (neighbor_list[index,:3]).astype(int)
        x,y,z = new_voxel

        # statistic for voxel 
        voxel_beta = beta_image[x,y,z]
        voxel_var = var_image[x,y,z]
        voxel_t = voxel_beta/np.sqrt(voxel_var)
        voxel = np.vstack((voxel,np.array([voxel_beta,voxel_var,voxel_t])))

        # compute statistic for region
        region_beta,region_var,region_t = model(voxel[:,0],voxel[:,1])
        print voxel_beta,voxel_var,region_t
        region = np.vstack((region,np.array([region_beta,region_var,region_t])))
        #seed_value = np.mean(voxel[:, 2])
        
        flag_image[x,y,z] = 2  
        region_t_image[x,y,z] = voxel_t
        neighbor_list =  np.delete(neighbor_list,(index),axis=0)
        #print '------------------------'
        #print  neighbor_list
        
        neighbor = np.tile(new_voxel,(nbsize,1)) + offsets

        nbx = np.all([inside(neighbor,image_shape),\
                   var_image[neighbor[:,0],neighbor[:,1],neighbor[:,2]] != 0 ,\
                   flag_image[neighbor[:,0],neighbor[:,1],neighbor[:,2]] == 0],axis = 0)
        #print nbx
        neighbor = neighbor[nbx,:]
        neighbor_t = beta_image[neighbor[:,0],neighbor[:,1],neighbor[:,2]]/ \
                np.sqrt(var_image[neighbor[:,0],neighbor[:,1],neighbor[:,2]])

        newnb  = np.hstack((neighbor,neighbor_t.reshape(-1,1)))
        neighbor_list = np.vstack((neighbor_list,newnb))
        flag_image[neighbor[:,0],neighbor[:,1],neighbor[:,2]] = 1
        
        t_change = region[-1,2] - region[-2,2]
         
    return  region,voxel   


def inside(coors,image_shape):
    if len(coors.shape) == 1: 
        coors = coors.reshape(1,3)
    
    return  np.all([coors[:,0] >= 0, coors[:,0] < image_shape[0], \
            coors[:,1] >= 0, coors[:,1] < image_shape[1], \
            coors[:,2] >= 0, coors[:,2] < image_shape[2]],axis = 0)


def fe(voxel_beta,voxel_var):
    beta = np.mean(voxel_beta)
    var  = np.sum(voxel_var)/(voxel_var.shape[0])**2
    t = beta/np.sqrt(var) 
    
    return beta,var,t
    
    
def re(voxel_beta,voxel_var):
    beta = np.mean(voxel_beta)
    N = voxel_beta.shape[0]
    var = np.var(voxel_beta)
    #print voxel_beta
    t = beta/np.sqrt(var,dtype=np.float64) 
    
    return beta,var,t

def me(voxel_beta,voxel_var):
    beta = np.mean(voxel_beta)
    N = voxel_beta.shape[0]
    var = (np.var(voxel_beta) + np.mean(voxel_var))/N  
    t = beta/np.sqrt(var) 
    
    return beta,var,t
