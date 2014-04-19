
import nibabel as nib
import matplotlib.pyplot as plt
import region_grow_froi as rgf
import numpy as np


beta_index = ["data/S1/cope1.nii.gz","data/S2/cope1.nii.gz","data/S3/cope1.nii.gz"]
var_index = ["data/S1/varcope1.nii.gz","data/S2/varcope1.nii.gz","data/S3/varcope1.nii.gz"]
t_index = ["data/S1/tstat1.nii.gz","data/S2/tstat1.nii.gz","data/S3/tstat1.nii.gz"]
seeds=[(26,38,25),(24,37,24),(24,37,25)]
output=['S1_FFA(26,38,25)','S2_FFA(24,37,24)','S3_FFA(24,37,25)']

for i in range(0,1):
    beta_img=nib.load(beta_index[i]) 
    var_img=nib.load(var_index[i])
    t_img=nib.load(t_index[i])       #provide data_head for storing the results

    beta_data=beta_img.get_data()
    var_data=var_img.get_data()
    t_data=t_img.get_data()
    region,voxel  = rgf.region_growing_froi(seeds[i],beta_data,var_data,'fe',6)
    
    #for i in range(10):
        #print region[i,2]

    plt.figure(1)
    plt.subplot(411)
    plt.plot(region[:,0])
    plt.ylabel('region beta')

    plt.subplot(412)
    plt.plot(region[:,1])
    plt.ylabel('region variance')

    plt.subplot(413)
    plt.plot(region[:,2])
    plt.ylabel('region t')    

    plt.subplot(414)
    plt.plot(np.diff(region[:,2]))
    plt.ylabel('region t change')   
    plt.show()


