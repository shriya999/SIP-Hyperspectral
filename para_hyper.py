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
re_grnd= grnd.ReadAsArray()

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
ground=np.reshape(re_grnd,(1,-1))
ground=np.vstack((ground,rn,cn))   # 1D array form of ground file at 220th row containing class numbers
 
#np.random.seed(0)                   
indices=np.random.permutation(len(mat.T))
part=int(0.8*new_cols)
train_ds=mat[:,indices[:part]]
test_ds=mat[:,indices[part:]]
ground_train=ground[:,indices[:part]]
ground_test=ground[:,indices[part:]]
test_cols=new_cols-part
train_cols=part


# location of ground file of test data is last row of test_ds i.e. index "bands"

pred=np.zeros((bands,test_cols))
for row in range(bands):
	sep_train = {}                         # calculating the summary of data for each band of channel individually
	for i in range(train_cols):
		vector = ground_train[0,i]              # keys of the dictionary are class numbers and values are lists of pixels with the 
		if (vector not in sep_train):                        # same class numbers for that row 
			sep_train[vector] = []
		sep_train[vector].append(train_ds[row,i])

	summary={}
	for j in sep_train.keys():                    # dictionary with keys as class numbers and corresponding mean and std deviation for that band
		class_mean=np.mean(sep_train[j])
		class_stdev=np.std(sep_train[j])
		lower=class_mean-3*class_stdev
		upper=class_mean+3*class_stdev
		summary[j]=[lower,upper]              # defining limits for the parallelepiped

	for k in range(test_cols):
		val=test_ds[row,k]
		for i in summary.keys():
			if val>summary[i][0] and val<summary[i][1]:
				pred[row,k]=int(i)
 
# overlapped pixels are assigned latest class and unclassified have class number as 0
	                                                                               
output=np.zeros(test_cols,)
for i in range(test_cols):
	counts = np.bincount(pred[:,i].astype(int))
	output[i]=np.argmax(counts)

for i in range(test_cols):
	if ground_test[0][i]==0:
		output[i]=0

count=0
for i in range(test_cols):
	if output[i]==ground_test[0][i]:
		count+=1

print "Accuracy: ",(count*100.0)/test_cols

# for plotting
out= np.zeros((rows,cols))

for i in range(new_cols -part):                                                   # for test data
	out[int(ground_test[1,i]),int(ground_test[2,i])]=int(output[i])
for i in range(part):
	out[int(ground_train[1,i]),int(ground_train[2,i])]=int(ground_train[0,i])  # class numbers of training data acc original indices

				
endtime=time.time()
print 'the program took '+str(endtime-starttime)+' seconds'

from matplotlib import pyplot as plt
plt.imshow(out, interpolation='nearest')
plt.show()


          

