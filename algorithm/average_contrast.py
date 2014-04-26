
import numpy as np
from region_growing import RegionGrowing
from connectivity import compute_offsets
from whetherinside import inside

class average_contrast(RegionGrowing):
    """
    Maximum average contrast region growing.
    """
    def contrast_max(image,seed,Num):
        """
        find the max average contrast point.
        """
        x,y,z = seed
        image_shape = image.shape
        if inside(seed,image_shape)!=True:
            print "The seed is out of the image range."
            return False

        contrast = []
        region_size = 1
        origin_t = image[x,y,z]
        inner_list = [origin_t]
        tmp_image = np.zeros_like(image)

        neighbor_free = 10000
        neighbor_pos = -1
        neighbor_list = np.zeros((neighbor_free,4))

        while region_size <= Num:
            for i in range(26):
                set0,set1,set2 = compute_offsets(3,26)[i]
                xn,yn,zn = x+set0,y+set1,z+set2
                if inside((xn,yn,zn),image_shape) and tmp_image[xn,yn,zn]==0:
                    neighbor_pos = neighbor_pos+1
                    neighbor_list[neighbor_pos] = [xn,yn,zn,image[xn,yn,zn]]
                    tmp_image[xn,yn,zn] = 1

            out_boundary = neighbor_list[np.nonzero(neighbor_list[:,3]),3]
            contrast = contrast + [np.mean(np.array(inner_list)) - np.mean(out_boundary)]

            tmp_image[x,y,z] = 2
            region_size += 1

            if (neighbor_pos+100 > neighbor_free):
                neighbor_free +=10000
                new_list = np.zeros((10000,4))
                neighbor_list = np.vstack((neighbor_list,new_list))
                #if the longth of neighbor_list is not enough,add another 10000

            distance = np.abs(neighbor_list[:neighbor_pos+1,3] - np.tile(origin_t,neighbor_pos+1))
            index = distance.argmin()
            x,y,z = neighbor_list[index][:3]
            inner_list = inner_list + [image[x,y,z]]
            neighbor_list[index] = neighbor_list[neighbor_pos]
            neighbor_pos -= 1

        return contrast,np.array(contrast).argmax()+1


