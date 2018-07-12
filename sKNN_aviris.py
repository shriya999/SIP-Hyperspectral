import gdal
import time
import sys
import random
import numpy as np
from operator import itemgetter
from gdalconst import *


print(sys.argv)         # prints filename as it is the first argument given to console
starttime=time.time()

ds =gdal.Open('/home/shriya/Dropbox/siplab/aviris.tif')
inimage = ds.ReadAsArray()
grnd = gdal.Open('/home/shriya/Dropbox/siplab/ground.tif')
ground= grnd.ReadAsArray()

cols = ds.RasterXSize           #145
rows = ds.RasterYSize		#145
bands = ds.RasterCount		#220

new_cols=rows*cols
new_rows=bands+3                            

rn=np.zeros((new_cols,))
cn=np.zeros((new_cols,))
count=0
for i in range(rows):
    for j in range(cols):
        rn[count]= i             # row numbers at 222
        cn[count]= j             # column numbers at 223
	count+=1             

mat=np.reshape(inimage,(bands,-1))    # conversion from 3D to 2D with shape 223*21025
re_grnd=np.reshape(ground,(1,-1))
mat=np.vstack((mat,re_grnd,rn,cn))   # 1D array form of ground file at 220th row containing class numbers


np.random.shuffle(mat.T)              # as random.shuffle only permutes the columns, transpose will shuffle only the columns

train_ds=mat[:,:int(0.7*new_cols)]         # partioning dataset into testing and training data based on a random ratio
test_ds=mat[:,int(0.7*new_cols):]
train_cols=int(0.7*new_cols)
test_cols=new_cols-train_cols

print "Input the value of K:"
k=input()
out= np.zeros((rows,cols))
       
for i in range(test_cols):
	mat_dist=np.zeros((train_cols,2))
	for j in range(train_cols):
		dist=np.sqrt(np.sum((train_ds[:new_rows,j]-test_ds[:new_rows,i])**2))
		mat_dist[j,0]=dist			        # storing distances
		mat_dist[j,1]=mat[bands,j]		        # storing class numbers
	new_dist=mat_dist[np.argsort(mat_dist[:,0])]                         # to sort by distances                                     
	counter= np.zeros((17,))
	for x in range(k):
		counter[int(new_dist[x][1])]+=1
	class_index = np.argmax(counter)                        # check case of np.count_nonzero(counter == max(counter))>1:
	out[int(test_ds[bands+1,i]),int(test_ds[bands+2,i])] =int(class_index)  

for i in range(0,rows):
    for j in range(0,cols):
        if ground[i,j]==0:
            out[i,j]=0
	    
for i in range(train_cols):
	out[int(train_ds[bands+1,i]),int(train_ds[bands+2,i])]=int(train_ds[bands,i])  # class numbers of training data acc original indices
	                                                                               
c=0
for i in range(rows):
	for j in range(cols):
        	if out[i,j]==ground[i,j]:
            		c+=1
            

print c
Accuracy = ((c-train_cols)*100.0)/test_cols                  # as training data shouldnt be used for accuracy calculations
print Accuracy
endtime=time.time()
print 'the program took '+str(endtime-starttime)+' seconds'

          
from matplotlib import pyplot as plt
plt.imshow(out, interpolation='nearest')
plt.show()

