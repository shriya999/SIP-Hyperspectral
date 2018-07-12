import gdal
import time
import sys
import random
import numpy as np
from sklearn.neural_network import MLPClassifier
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
new_rows=bands                            

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

mat= mat.astype(float)
ground= ground.astype(float)      # as standard scaler takes only float values as inputs

np.random.seed(0)               # to produce the same set of random numbers each time
indices=np.random.permutation(len(mat.T))
part=int(0.7*new_cols)
mat_train=mat[:,indices[:part]]
mat_test=mat[:,indices[part:]]
ground_train=ground[:,indices[:part]]
ground_test=ground[:,indices[part:]]

from sklearn.preprocessing import StandardScaler   # difficulty converging before the max iterations if data isnt normalized
scaler = StandardScaler()                          # Fit only to the training data
scaler.fit(mat_train.T)
X_train= scaler.transform(mat_train.T)
X_test= scaler.transform(mat_test.T)

clf= MLPClassifier()                          # default: one hidden layer with 100 neurons, lr=0.001, epochs=200                               
clf.fit(X_train , ground_train[0,:])
output=clf.predict(X_test)     

for i in range(new_cols-part):
	if ground_test[0,i]==0:
            output[i]=0

c=0
for i in range(new_cols-part):
        if output[i]==ground_test[0,i]:
            	c+=1

Accuracy = (c*100.0)/(new_cols-part)                  # as training data shouldnt be used for accuracy calculations
print Accuracy

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

