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

inimage= scipy.io.loadmat('/home/shriya/Dropbox/siplab/SalinasA_corrected.mat') # 6 classes
grnd= scipy.io.loadmat('/home/shriya/Dropbox/siplab/SalinasA_gt.mat')

ds=inimage['salinasA_corrected']
ground=grnd['salinasA_gt']

band=ds[:,:,0:2]                  # obtaining first two bands only

mat=np.reshape(band,(2,-1))              # conversion from 3D to 2D with shape 221*21025
re_grnd=np.reshape(ground,(1,-1))
mat=np.vstack((mat,re_grnd))

(rows,cols)=mat.shape

ratio=int(0.67*cols)
train_ds=mat[:,:ratio]         # partioning dataset into testing and training data based on a random ratio
test_ds=mat[:,ratio:]
train_cols=ratio
test_cols=cols-ratio

pred=np.zeros((2,test_cols))
out=np.zeros((test_cols,))

# FOR BAND1

sep_train = {}
summary={0:[],1:[],10:[],11:[],12:[],13:[],14:[]}                         # calculating the summary of data for each band of channel individually

for i in range(train_cols):
	vector = train_ds[2,i]              # keys of the dictionary are class numbers and values are lists of pixels with the 
	if (vector not in sep_train):                        # same class numbers for that row 
		sep_train[vector] = []
	sep_train[vector].append(train_ds[0,i])

# for class 0                   # dictionary with keys as class numbers and corresponding mean and std deviation for that band
class_mean=np.mean(sep_train[0])
class_stdev=np.std(sep_train[0])
lower=class_mean-1*class_stdev
upper=class_mean+1*class_stdev
summary[0]=[lower,upper]              # defining limits for the parallelepiped

# for class 1                  # dictionary with keys as class numbers and corresponding mean and std deviation for that band
class_mean=np.mean(sep_train[1])
class_stdev=np.std(sep_train[1])
lower=class_mean-1*class_stdev
upper=class_mean+1*class_stdev
summary[1]=[lower,upper]              # defining limits for the parallelepiped

# for class 10                   # dictionary with keys as class numbers and corresponding mean and std deviation for that band
class_mean=np.mean(sep_train[10])
class_stdev=np.std(sep_train[10])
lower=class_mean-1*class_stdev
upper=class_mean+1*class_stdev
summary[10]=[lower,upper]              # defining limits for the parallelepiped

# for class 11                   # dictionary with keys as class numbers and corresponding mean and std deviation for that band
class_mean=np.mean(sep_train[11])
class_stdev=np.std(sep_train[11])
lower=class_mean-1*class_stdev
upper=class_mean+1*class_stdev
summary[11]=[lower,upper]              # defining limits for the parallelepiped

# for class 12                   # dictionary with keys as class numbers and corresponding mean and std deviation for that band
class_mean=np.mean(sep_train[12])
class_stdev=np.std(sep_train[12])
lower=class_mean-1*class_stdev
upper=class_mean+1*class_stdev
summary[12]=[lower,upper]              # defining limits for the parallelepiped

# for class 13                   # dictionary with keys as class numbers and corresponding mean and std deviation for that band
class_mean=np.mean(sep_train[13])
class_stdev=np.std(sep_train[13])
lower=class_mean-2*class_stdev
upper=class_mean+2*class_stdev
summary[13]=[lower,upper]              # defining limits for the parallelepiped

# for class 14                   # dictionary with keys as class numbers and corresponding mean and std deviation for that band
class_mean=np.mean(sep_train[14])
class_stdev=np.std(sep_train[14])
lower=class_mean-1*class_stdev
upper=class_mean+1*class_stdev
summary[14]=[lower,upper]              # defining limits for the parallelepiped


for k in range(test_cols):
	val=test_ds[0,k]
	for i in summary.keys():
		if val>summary[i][0] and val<summary[i][1]:
			pred[0,k]=int(i)

# FOR BAND2

sep_train = {}
summary={0:[],1:[],10:[],11:[],12:[],13:[],14:[]}                         # calculating the summary of data for each band of channel individually

for i in range(train_cols):
	vector = train_ds[2,i]              # keys of the dictionary are class numbers and values are lists of pixels with the 
	if (vector not in sep_train):                        # same class numbers for that row 
		sep_train[vector] = []
	sep_train[vector].append(train_ds[0,i])

# for class 0                   # dictionary with keys as class numbers and corresponding mean and std deviation for that band
class_mean=np.mean(sep_train[0])
class_stdev=np.std(sep_train[0])
lower=class_mean-1*class_stdev
upper=class_mean+1*class_stdev
summary[0]=[lower,upper]              # defining limits for the parallelepiped

# for class 1                  # dictionary with keys as class numbers and corresponding mean and std deviation for that band
class_mean=np.mean(sep_train[1])
class_stdev=np.std(sep_train[1])
lower=class_mean-1*class_stdev
upper=class_mean+1*class_stdev
summary[1]=[lower,upper]              # defining limits for the parallelepiped

# for class 10                   # dictionary with keys as class numbers and corresponding mean and std deviation for that band
class_mean=np.mean(sep_train[10])
class_stdev=np.std(sep_train[10])
lower=class_mean-1*class_stdev
upper=class_mean+1*class_stdev
summary[10]=[lower,upper]              # defining limits for the parallelepiped

# for class 11                   # dictionary with keys as class numbers and corresponding mean and std deviation for that band
class_mean=np.mean(sep_train[11])
class_stdev=np.std(sep_train[11])
lower=class_mean-1*class_stdev
upper=class_mean+1*class_stdev
summary[11]=[lower,upper]              # defining limits for the parallelepiped

# for class 12                   # dictionary with keys as class numbers and corresponding mean and std deviation for that band
class_mean=np.mean(sep_train[12])
class_stdev=np.std(sep_train[12])
lower=class_mean-1*class_stdev
upper=class_mean+1*class_stdev
summary[12]=[lower,upper]              # defining limits for the parallelepiped

# for class 13                   # dictionary with keys as class numbers and corresponding mean and std deviation for that band
class_mean=np.mean(sep_train[13])
class_stdev=np.std(sep_train[13])
lower=class_mean-2*class_stdev
upper=class_mean+2*class_stdev
summary[13]=[lower,upper]              # defining limits for the parallelepiped

# for class 14                   # dictionary with keys as class numbers and corresponding mean and std deviation for that band
class_mean=np.mean(sep_train[14])
class_stdev=np.std(sep_train[14])
lower=class_mean-1*class_stdev
upper=class_mean+1*class_stdev
summary[14]=[lower,upper]              # defining limits for the parallelepiped

for k in range(test_cols):
	val=test_ds[1,k]
	for i in summary.keys():
		if val>summary[i][0] and val<summary[i][1]:
			pred[1,k]=int(i)
 
# overlapped pixels are assigned latest class and unclassified have class number as 0

for i in range(test_cols):
	if pred[0,i]==pred[1,i]:
		out[i]=pred[0,i]

	                                                                              
for i in range(test_cols):
	if test_ds[2,i]==0:
		out[i]=0

count=0
for i in range(test_cols):
	if out[i]==test_ds[2,i]:
		count+=1

print count
print "Accuracy: ",(count*100.0)/test_cols
				
endtime=time.time()
print 'the program took '+str(endtime-starttime)+' seconds'

          

