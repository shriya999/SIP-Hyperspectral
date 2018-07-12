import gdal
import time
import sys
import random
import numpy as np
from operator import itemgetter
from gdalconst import *
import scipy.io

print(sys.argv)         # prints filename as it is the first argument given to console
starttime=time.time()

ds= scipy.io.loadmat('/home/shriya/Dropbox/siplab/SalinasA_corrected.mat')
grnd= scipy.io.loadmat('/home/shriya/Dropbox/siplab/SalinasA_gt.mat')

inimage=ds['salinasA_corrected']
ground=grnd['salinasA_gt']

(rows,cols,bands)=inimage.shape

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

train_cols=int(0.8*new_cols)
train_ds=mat[:,:train_cols]         # partioning dataset into testing and training data based on a random ratio
test_ds=mat[:,train_cols:]
test_cols=new_cols-train_cols

k=input("Input the value of K:")
out= np.zeros((rows,cols))

exclusion=0       
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
	
	if np.count_nonzero(counter == max(counter))>1:
		class_index=-1
		exclusion+=1                                    # cases of more than one class with max occurence is excluded
	else:
		class_index = np.argmax(counter)
	
	out[int(test_ds[bands+1,i]),int(test_ds[bands+2,i])] =int(class_index)  # storing class numbers for test data 

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
Accuracy = (c*100.0)/((rows*cols)-exclusion)
print Accuracy
endtime=time.time()
print 'the program took '+str(endtime-starttime)+' seconds'

          
from matplotlib import pyplot as plt
plt.imshow(out, interpolation='nearest')
#plt.imshow(ground,interpolation='nearest')
plt.show()

