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

mat= mat.astype(float)
ground= ground.astype(float)      # as standard scaler takes only float values as inputs

np.random.seed(0)
indices=np.random.permutation(len(mat.T))
part=int(0.7*new_cols)
mat_train=mat[:,indices[:part]]
mat_test=mat[:,indices[part:]]
Y_train=ground[:,indices[:part]]
Y_test=ground[:,indices[part:]]
train_cols=part
test_cols=new_cols-part

from sklearn.preprocessing import StandardScaler   # difficulty converging before the max iterations if data isnt normalized
scaler = StandardScaler()                          # Fit only to the training data using tranpose
scaler.fit(mat_train.T)
mat_train= scaler.transform(mat_train.T)
mat_test= scaler.transform(mat_test.T)

X_train=mat_train.T                                # to get back original shape
X_test=mat_test.T


def activ(x):
	return 1/(1+np.exp(-x))
def activ_deriv(x):
	return activ(x)*(1-activ(x))
 
# layer 0 is input,1 is hidden and 2 is output
layers=[bands,10,16]
     
hidden_weights = [[random.random() for input_node in range(layers[0] + 1)] for hidden_node in range(layers[1])] # shape is 10 rows 221 columns
output_weights= [[random.random() for hidden_node in range(layers[1] + 1)] for output_node in range(layers[2])] # shape is 16 rows 10 columns
hidden_weights=np.array(hidden_weights,dtype=np.float128)
output_weights=np.array(output_weights,dtype=np.float128)

bias=np.zeros((1,X_train.shape[1]))
X_train=np.append(X_train,bias,axis=0)
epochs=10
lr=0.2

# for training data
 
for m in range(epochs):              
	for i in range(train_cols):
		inputs=X_train[:,i]                               # all bands corresponding to that pixel are appended to inputs.
		inputs=np.array(inputs,dtype=np.float128)
		# forward propagation
		new_inputs=[]                                        # for hidden layer   
		for j in range(layers[1]):
			new_inputs.append(activ(np.dot(inputs[:],hidden_weights[j])))   # output signal1
		new_inputs.append(1)
		new_inputs=np.array(new_inputs,dtype=np.float128)

		outputs=[]                                            # for output layer
		for j in range(layers[2]):
			outputs.append(activ(np.dot(new_inputs[:],output_weights[j])))   # output signal1

		y=[0 for k in range(16)]
		for j in range(16):
			if j==Y_train[0][i]:
				y[j]=1

		# back propagation
		output_delta=[]                                      # delta for output layer
		for j in range(layers[2]):
			error=(y[j]-outputs[j])
			output_delta.append(error*activ_deriv(outputs[j]))	

		hidden_delta=[]                                       # delta for hidden layer
		for j in range(layers[1]):
			weighted_sum=np.dot(output_delta[:],output_weights.T[j])
			hidden_delta.append(activ_deriv(new_inputs[j])*weighted_sum)

		for j in range(layers[2]):                            # updating for output layer
			for k in range(layers[1]+1):
				output_weights[j][k]+=new_inputs[k]*lr*output_delta[j]

		for j in range(layers[1]):                            # updating for hidden layer
			for k in range(layers[0]+1):
				hidden_weights[j][k]+=inputs[k]*lr*hidden_delta[j]

# for testing data

output=[0 for j in range(test_cols)]
bias=np.zeros((1,X_test.shape[1]))
X_test=np.append(X_test,bias,axis=0)

for i in range(test_cols):
	inputs=X_test[:,i]                               # all bands corresponding to that pixel are appended to inputs.
	inputs=np.array(inputs,dtype=np.float128)
	
	new_inputs=[]                                        # for hidden layer   
	for j in range(layers[1]):
		new_inputs.append(activ(np.dot(inputs[:],hidden_weights[j])))   # output signal1
	new_inputs.append(1)
	new_inputs=np.array(new_inputs,dtype=np.float128)

	outputs=[]                                            # for output layer
	for j in range(layers[2]):
		outputs.append(activ(np.dot(new_inputs[:],output_weights[j])))   # output signal1
	output[i]=outputs.index(max(outputs))

# accuracy calculations

for i in range(test_cols):
	if Y_test[0,i]==0:
            output[i]=0

c=0
for i in range(test_cols):
        if output[i]==Y_test[0,i]:
            	c+=1

Accuracy = (c*100.0)/(test_cols)                  # as training data shouldnt be used for accuracy calculations
print Accuracy
endtime=time.time()
print 'the program took '+str(endtime-starttime)+' seconds'

# plotting output

out= np.zeros((rows,cols))

for i in range(test_cols):                                                   # for test data
	out[int(Y_test[1,i]),int(Y_test[2,i])]=int(output[i])
for i in range(train_cols):
	out[int(Y_train[1,i]),int(Y_train[2,i])]=int(Y_train[0,i])  # class numbers of training data acc original indices
          
from matplotlib import pyplot as plt
plt.imshow(out, interpolation='nearest')
plt.show()	
			



