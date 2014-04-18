class Queue:                              #define a class Queue
    def __init__(self):             #initialize a queue
        self.items = []
 
    def isEmpty(self):                    # decide whether it is an empty queue
        return self.items==[]
 
    def enque(self,item):                 #insert (index,object)
        self.items.insert(0,item)
 
    def deque(self):                      #pop the queue
        return self.items.pop()
 
    def qsize(self):                      #return the size of queue
        return len(self.items)
    
    def isInside(self, item):              #decide whether it is in the queue
        return (item in self.items)
        
 
import Image,os  
import nibabel as nib
import numpy as np


img=nib.load("MNI152_T1_2mm_brain.nii")                        #import the module Image and os
data=img.get_data()
temp=data[50,:,:]
 
for i in range(0,109):
    for j in range(0,91):
        temp[i][j]=temp[i][j]/8339.0*256

image=Image.fromarray(temp)             

image1=image.resize((300,300),Image.NEAREST)
image1.show()                               #output the original image

 
def regiongrow(image,epsilon,start_point):            #define function(original image\gradient difference\start point)radient
 
    Q = Queue()                                       #class Q
    s = []                                            #list of new picture point
    
    x = start_point[0]
    y = start_point[1]                                #(x,y) start point
    
    image = image.convert("L")                        #the function to transfer image to grey-scale map
    Q.enque((x,y))                                    #insert the point(x,y) of queue
 
    
    while not Q.isEmpty():                            #the circle when it is not empty
 
        t = Q.deque()
        x = t[0]
        y = t[1]
              #in the size of picture and the gradient difference is not so large  
        if x < image.size[0]-1 and \
           abs(  image.getpixel( (x + 1 , y) ) - image.getpixel( (x , y) )  ) <= epsilon :
 
            if not Q.isInside( (x + 1 , y) ) and not (x + 1 , y) in s: #the point is not in the new list s
                Q.enque( (x + 1 , y) )                                 #then insert the point
 
                
        if x > 0 and \
           abs(  image.getpixel( (x - 1 , y) ) - image.getpixel( (x , y) )  ) <= epsilon:
 
            if not Q.isInside( (x - 1 , y) ) and not (x - 1 , y) in s:
                Q.enque( (x - 1 , y) )
 
                     
        if y < (image.size[1] - 1) and \
           abs(  image.getpixel( (x , y + 1) ) - image.getpixel( (x , y) )  ) <= epsilon:
 
            if not Q.isInside( (x, y + 1) ) and not (x , y + 1) in s:
                Q.enque( (x , y + 1) )
 
                    
        if y > 0 and \
           abs(  image.getpixel( (x , y - 1) ) - image.getpixel( (x , y) )  ) <= epsilon:
 
            if not Q.isInside( (x , y - 1) ) and not (x , y - 1) in s:
                Q.enque( (x , y - 1) )
 
 
        if t not in s:
            s.append( t )             #affix the start point 
 
            
    image.load()                       #reload the image
    putpixel = image.im.putpixel       #output the pixel point
    
    for i in range ( image.size[0] ):
        for j in range ( image.size[1] ):
            putpixel( (i , j) , 0 )
 
    for i in s:
        putpixel(i , 150)
        
    output=raw_input("enter save file name : ")
    image.thumbnail( (image.size[0] , image.size[1]) , Image.ANTIALIAS )#set the size of picture,and the blur quantity

    image.save(output + ".JPEG" , "JPEG")

    image=image.resize((300,300),Image.NEAREST)
    image.show(output + ".JPEG")

regiongrow(image,7,(50,50))     #use the function


