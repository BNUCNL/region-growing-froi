
import nibabel as nib
import matplotlib.pyplot as plt
import region_grow_froi as rgf
import numpy as np


beta_index = ["/nfs/t2/fmricenter/volume/S0019/obj.gfeat/cope1.feat/stats/cope1.nii.gz",\
              "/nfs/t2/fmricenter/volume/S0023/obj.gfeat/cope1.feat/stats/cope1.nii.gz",\
              "/nfs/t2/fmricenter/volume/S0023/obj.gfeat/cope1.feat/stats/cope1.nii.gz",\
              "/nfs/t2/fmricenter/volume/S0025/obj.gfeat/cope1.feat/stats/cope1.nii.gz",\
              "/nfs/t2/fmricenter/volume/S0027/obj.gfeat/cope1.feat/stats/cope1.nii.gz",\
              "/nfs/t2/fmricenter/volume/S0053/obj.gfeat/cope1.feat/stats/cope1.nii.gz",\
              "/nfs/t2/fmricenter/volume/S0057/obj.gfeat/cope1.feat/stats/cope1.nii.gz",\
              "/nfs/t2/fmricenter/volume/S0012/obj.gfeat/cope1.feat/stats/cope1.nii.gz",\
              "/nfs/t2/fmricenter/volume/S0014/obj.gfeat/cope1.feat/stats/cope1.nii.gz",\
              "/nfs/t2/fmricenter/volume/S0016/obj.gfeat/cope1.feat/stats/cope1.nii.gz",\
              "/nfs/t2/fmricenter/volume/S0020/obj.gfeat/cope1.feat/stats/cope1.nii.gz",\
              "/nfs/t2/fmricenter/volume/S0031/obj.gfeat/cope1.feat/stats/cope1.nii.gz",\
              "/nfs/t2/fmricenter/volume/S0034/obj.gfeat/cope1.feat/stats/cope1.nii.gz",\
              "/nfs/t2/fmricenter/volume/S0035/obj.gfeat/cope1.feat/stats/cope1.nii.gz",\
              "/nfs/t2/fmricenter/volume/S0036/obj.gfeat/cope1.feat/stats/cope1.nii.gz",\
              "/nfs/t2/fmricenter/volume/S0037/obj.gfeat/cope1.feat/stats/cope1.nii.gz",\
              "/nfs/t2/fmricenter/volume/S0038/obj.gfeat/cope1.feat/stats/cope1.nii.gz",\
              "/nfs/t2/fmricenter/volume/S0040/obj.gfeat/cope1.feat/stats/cope1.nii.gz",\
              "/nfs/t2/fmricenter/volume/S0041/obj.gfeat/cope1.feat/stats/cope1.nii.gz",\
              "/nfs/t2/fmricenter/volume/S0051/obj.gfeat/cope1.feat/stats/cope1.nii.gz"]

var_index = ["/nfs/t2/fmricenter/volume/S0019/obj.gfeat/cope1.feat/stats/varcope1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0023/obj.gfeat/cope1.feat/stats/varcope1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0023/obj.gfeat/cope1.feat/stats/varcope1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0025/obj.gfeat/cope1.feat/stats/varcope1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0027/obj.gfeat/cope1.feat/stats/varcope1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0053/obj.gfeat/cope1.feat/stats/varcope1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0057/obj.gfeat/cope1.feat/stats/varcope1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0012/obj.gfeat/cope1.feat/stats/varcope1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0014/obj.gfeat/cope1.feat/stats/varcope1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0016/obj.gfeat/cope1.feat/stats/varcope1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0020/obj.gfeat/cope1.feat/stats/varcope1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0031/obj.gfeat/cope1.feat/stats/varcope1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0034/obj.gfeat/cope1.feat/stats/varcope1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0035/obj.gfeat/cope1.feat/stats/varcope1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0036/obj.gfeat/cope1.feat/stats/varcope1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0037/obj.gfeat/cope1.feat/stats/varcope1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0038/obj.gfeat/cope1.feat/stats/varcope1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0040/obj.gfeat/cope1.feat/stats/varcope1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0041/obj.gfeat/cope1.feat/stats/varcope1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0051/obj.gfeat/cope1.feat/stats/varcope1.nii.gz"]

t_index = ["/nfs/t2/fmricenter/volume/S0019/obj.gfeat/cope1.feat/stats/tstat1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0023/obj.gfeat/cope1.feat/stats/tstat1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0023/obj.gfeat/cope1.feat/stats/tstat1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0025/obj.gfeat/cope1.feat/stats/tstat1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0027/obj.gfeat/cope1.feat/stats/tstat1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0053/obj.gfeat/cope1.feat/stats/tstat1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0057/obj.gfeat/cope1.feat/stats/tstat1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0012/obj.gfeat/cope1.feat/stats/tstat1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0014/obj.gfeat/cope1.feat/stats/tstat1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0016/obj.gfeat/cope1.feat/stats/tstat1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0020/obj.gfeat/cope1.feat/stats/tstat1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0031/obj.gfeat/cope1.feat/stats/tstat1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0034/obj.gfeat/cope1.feat/stats/tstat1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0035/obj.gfeat/cope1.feat/stats/tstat1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0036/obj.gfeat/cope1.feat/stats/tstat1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0037/obj.gfeat/cope1.feat/stats/tstat1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0038/obj.gfeat/cope1.feat/stats/tstat1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0040/obj.gfeat/cope1.feat/stats/tstat1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0041/obj.gfeat/cope1.feat/stats/tstat1.nii.gz",\
             "/nfs/t2/fmricenter/volume/S0051/obj.gfeat/cope1.feat/stats/tstat1.nii.gz"]

seeds=[(25,37,26),(26,36,24),(63,37,24),(68,36,24),(27,30,26),(23,40,27),(25,40,24),(25,41,26),(24,35,22),(22,35,24),(27,29,28),(23,38,28),(23,37,24),(26,38,25),(24,38,27),(22,31,26),(23,34,26),(24,37,24),(24,37,25),(22,38,27)]

output=['S0019_FFA(25,41,25)','S0023_FFA(26,36,24)','S0023_lFFA(63,37,24)','S0025_lFFA(68,36,24)','S0027_FFA(27,30,26)','S0053_FFA(23,40,27)','S0057_FFA(25,40,24)','S0012_FFA(25,41,26)','S0014_FFA(24,35,22)','S0016_FFA(22,35,24)','S0020_FFA(27,29,28)','S0031_FFA(23,38,28)','S0034_FFA(23,37,24)','S0035_FFA(26,38,25)','S0036_FFA(24,38,27)','S0037_FFA(22,31,26)','S0038_FFA(23,34,26)','S0040_FFA(24,37,24)','S0041_FFA(24,37,25)','S0051_FFA(22,38,27)']


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


