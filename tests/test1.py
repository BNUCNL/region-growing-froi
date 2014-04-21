
import nibabel as nib
import matplotlib.pyplot as plt
import region_grow_froi1


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

    region,num,t_list,t_change_list = region_grow_froi1.region_growing(seeds[i],beta_data,var_data,6,'fixed',output[i])
    t_img._data = region
    y1 = t_list
    y2 = t_change_list

    #nib.save(t_img,output[i]+".nii.gz")
    print num
    
    plt.figure(1)
    plt.subplot(211)
    plt.plot(y1)
    plt.ylabel('region t value')

    plt.subplot(212)
    plt.plot(y2)
    plt.ylabel('region t change')
    plt.show()


    #x1 = range(1,num+2)
    #plt.plot(x1,y1,label='cluster t value')
    #x2 = range(2,num+2)
    #plt.plot(x2,y2,label='cluster t change')
    #x3 = range(0,num+10)
    #y3=[]
    #for i in x3:
        #y3 = y3 + [0]
    #plt.plot(x3,y3)
    #plt.xlabel('cluster size')
    #plt.ylabel('cluster t change')
    #plt.legend(loc='upper left')
    #plt.show()
