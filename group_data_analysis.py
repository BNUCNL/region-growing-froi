
import numpy as np
import matplotlib.pyplot as plt

N=1000
output=['S1(26,38,25)','S2(24,37,24)','S3(24,37,25)','S4(24,38,24)','S5(23,38,26)']

result = np.zeros((N,6))
for i in range(0,5):
    j=0
    with open(output[i],'r') as f:
        for line in f:
            #print map(float,line.split(','))[0]
            result[j][i] = map(float,line.split(','))[0]
            j=j+1
for k in range(N):
    result[k,5] = np.mean(result[k,0:5])
plt.plot(result[:,5])
plt.xlabel('voxels')
plt.ylabel('group average of average contrast')
plt.show()


