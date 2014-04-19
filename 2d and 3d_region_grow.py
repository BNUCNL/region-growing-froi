
import Image,os,sys
import nibabel as nib
import numpy as np


class Queue:                              #define a class Queue
    def __init__(self):             #initialize a queue
        self.items = []
 
    def isEmpty(self):                   # decide whether it is an empty queue
        return self.items==[]
 
    def enque(self,item):                 #insert the queue
        self.items.insert(0,item)
 
    def deque(self):                      #pop the queue
        return self.items.pop()
 
    def qsize(self):                      #return the size of queue
        return len(self.items)
    
    def isInside(self, item):              #decide whether it is in the queue
        return (item in self.items)


def getVoxel(x,y,z,data):                  #get the value of Voxel point
    if x<0 or x>shape[0]:
        return False

    if y<0 or y>shape[1]:
        return False

    if z<0 or z>shape[2]:
        return False

    return data[x,y,z]


def setVoxel(x,y,z,data,value):           #set the value of Voxel point
    if x<0 or x>shape[0]:
        return False

    if y<0 or y>shape[1]:
        return False

    if z<0 or z>shape[2]:
        return False

    data[x,y,z] = value
    return True        



def regiongrow2(image,epsilon,start_point):
    #define function(original image\gradient difference\start point)radient
 
    Q2 = Queue()                                       #class Q
    s2 = []                                            #list of new picture point
    
    x = start_point[0]
    y = start_point[1]                                #(x,y) start point
    
    image = image.convert("L")                        #the function to transfer image to grey-scale map
    Q2.enque((x,y))                                    #insert the point(x,y) of queue
 
    
    while not Q2.isEmpty():                            #the circle when it is not empty
 
        t = Q2.deque()
        x = t[0]
        y = t[1]
        
        #in the size of picture and the gradient difference is not so large
        
        if x < image.size[0]-1 and \
           abs(  image.getpixel( (x + 1 , y) ) - image.getpixel( (x , y) )  ) <= epsilon :
 
            if not Q2.isInside( (x + 1 , y) ) and not (x + 1 , y) in s2: 
                Q2.enque( (x + 1 , y) )
 
                
        if x > 0 and \
           abs(  image.getpixel( (x - 1 , y) ) - image.getpixel( (x , y) )  ) <= epsilon:
 
            if not Q2.isInside( (x - 1 , y) ) and not (x - 1 , y) in s2:
                Q2.enque( (x - 1 , y) )
 
                     
        if y < (image.size[1] - 1) and \
           abs(  image.getpixel( (x , y + 1) ) - image.getpixel( (x , y) )  ) <= epsilon:
 
            if not Q2.isInside( (x, y + 1) ) and not (x , y + 1) in s2:
                Q2.enque( (x , y + 1) )
 
                    
        if y > 0 and \
           abs(  image.getpixel( (x , y - 1) ) - image.getpixel( (x , y) )  ) <= epsilon:
 
            if not Q2.isInside( (x , y - 1) ) and not (x , y - 1) in s2:
                Q2.enque( (x , y - 1) )
 
 
        if t not in s2:
            s2.append( t )
            #affix the start point

    image.load()    
    putpixel = image.im.putpixel
              
    for i in range ( image.size[0] ):
        for j in range ( image.size[1] ):
            putpixel( (i , j) , 0 )
 
    for i in s2:
        putpixel(i , 150)
    return image




def regiongrow3(image,epsilon,start_point):           
    #define function(original image\gradient difference\start point)radient
 
    Q = Queue()                                      #Q is the queue of flag 1
    s = []                                           #s is the list of flag 2
    #flag=0
    
    x = start_point[0]
    y = start_point[1]
    z = start_point[2]
    
    Q.enque((x,y,z))                                    
 
    
    while not Q.isEmpty():                            
        #the circle when it is not empty
 
        t = Q.deque()
        x = t[0]
        y = t[1]
        z = t[2]
              
        if x < shape[0] and \
           abs(  getVoxel(x+1,y,z,image) - getVoxel(x,y,z,image)  ) <= epsilon :
 
            if not Q.isInside( (x + 1 , y ,z) ) and not (x + 1 , y ,z) in s:              
                #the point is not in the Q and s
                Q.enque( (x + 1 , y ,z) )                                 
                #then insert the point
 
                
        if x > 0 and \
           abs(  getVoxel(x-1,y,z,image) - getVoxel(x,y,z,image)  ) <= epsilon:
 
            if not Q.isInside( (x - 1 , y ,z) ) and not (x - 1 , y ,z) in s:
                Q.enque( (x - 1 , y ,z) )
 
                     
        if y < shape[1] and \
           abs(  getVoxel(x,y+1,z,image) - getVoxel(x,y,z,image)  ) <= epsilon:
 
            if not Q.isInside( (x, y + 1 ,z) ) and not (x , y + 1 ,z) in s:
                Q.enque( (x , y + 1 ,z) )
 
                    
        if y > 0 and \
           abs(  getVoxel(x,y-1,z,image) - getVoxel(x,y,z,image)  ) <= epsilon:
 
            if not Q.isInside( (x , y - 1 ,z) ) and not (x , y - 1 ,z) in s:
                Q.enque( (x , y - 1 ,z) )
 
        if z < shape[2] and \
           abs(  getVoxel(x,y,z+1,image) - getVoxel(x,y,z,image)  ) <= epsilon:
 
            if not Q.isInside( (x, y , z+1) ) and not (x, y , z+1) in s:
                Q.enque( (x, y , z+1) )
 
                    
        if z > 0 and \
           abs(  getVoxel(x,y,z-1,image) - getVoxel(x,y,z,image)  ) <= epsilon:
 
            if not Q.isInside( (x , y , z-1) ) and not (x , y , z-1) in s:
                Q.enque( (x , y , z-1) )

 
        if t not in s:
            s.append( t )             #affix the start point 
            #flag=flag+1
            
    
    for i in range ( 0,shape[0] ):
        for j in range ( 0,shape[1] ):
            for k in range (0,shape[2]):
                if not (i,j,k) in s:
                    setVoxel(i,j,k,image,0)
 




if __name__=='__main__':
    x=1
    while x!=0:
        print 'please input a integer:\n(0 means end, 2 means 2D grow, 3 means 3D grow)'
        x=int (raw_input(":"))

        if x==3:    
            img=nib.load("MNI152_T1_2mm_brain.nii")                        
            data=img.get_data()
            shape=data.shape

            print 'do you want to see slice and slice segmention'
            print 'please input a integer:\n(1 means yes, 0 means no)'
            y=int (raw_input(":"))
            
            if y==1:
                print 'choose whice slice to grow:'
                k=int (raw_input("x="))
                temp=data[k,:,:]
 
                for i in range(0,shape[1]):
                    for j in range(0,shape[2]):
                        temp[i][j]=temp[i][j]/8339.0*256

                image1=Image.fromarray(temp)             
                print 'this is the orignal image'
                #image1=image1.resize((300,300),Image.NEAREST)
                image1.show()
                
                print 'input 2D seed point'
                a=int (raw_input("x="))
                b=int (raw_input("y="))

                image1=regiongrow2(image1,7,(a,b))            
                #image=regiongrow2(image,7,(360,180))

                output2=raw_input("\nnow the 2D regiongrowing is end \nplease input your file name\n : ")    
                #image.thumbnail( (image.size[0] , image.size[1]) , Image.ANTIALIAS )
                #set the size of picture,and the blur quantity    
                image1.save(output2 + ".JPEG" , "JPEG")

                print 'this is the new image after growing:\n'
                image1.show(output2 + ".JPEG")


            else:
                print 'input 3D seed point'
                a=int (raw_input("x="))
                b=int (raw_input("y="))           
                c=int (raw_input("z="))

                regiongrow3(data,35,(a,b,c))            
                #regiongrow3(data,35,(50,50,50))   #use the function

                #img=nib.AnalyzeImage(data,np.eye(4))
                img._data=data

                output3=raw_input("\nnow the 3D regiongrowing is end \nplease input your file name\n :")
                nib.save(img,output3+".nii.gz")

                os.system("fslview %s.nii.gz" % output3)
                #print 'congratulations,now you have finished region growing'


        elif x==2:
            image=Image.open("grow.jpg")
            print 'this is the orignal image'
            image.show() 

            print 'input 2D seed point'
            a=int (raw_input("x="))
            b=int (raw_input("y="))

            image=regiongrow2(image,7,(a,b))            
            #image=regiongrow2(image,7,(360,180))

            output2=raw_input("\nnow the 2D regiongrowing is end \nplease input your file name\n : ")    
            #image.thumbnail( (image.size[0] , image.size[1]) , Image.ANTIALIAS )
            #set the size of picture,and the blur quantity    
            image.save(output2 + ".JPEG" , "JPEG")
            print 'this is the new image after growing:\n'

            image.show(output2 + ".JPEG")


        elif x!=0:
            print 'wrong number!\n'
        else:
            print "end"
            break

