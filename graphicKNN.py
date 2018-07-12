import gdal
import time
import sys
import random
import numpy as np
from operator import itemgetter
from gdalconst import *
from PyQt4 import QtCore
from PyQt4 import QtGui

class Window(QtGui.QMainWindow):
    def __init__(self):
        super(Window,self).__init__()

        self.container = QtGui.QWidget()                         # creates a widget for inserting buttons
        self.setCentralWidget(self.container)                    
        self.container_lay = QtGui.QVBoxLayout()
        self.container.setLayout(self.container_lay)
	self.setGeometry(100,100,400,300)
	self.setWindowTitle("GUI_KNN")
	
	self.btn1 = QtGui.QPushButton('Choose original file', self)
        self.container_lay.addWidget(self.btn1)
	self.btn2 = QtGui.QPushButton('Choose ground file', self)
        self.container_lay.addWidget(self.btn2)
	self.connect(self.btn1, QtCore.SIGNAL('clicked()'), self.get_fname1) 
	self.connect(self.btn2, QtCore.SIGNAL('clicked()'), self.get_fname2)

        # Input
        self.le = QtGui.QLineEdit()
	self.le.setPlaceholderText("Enter K value")
        self.container_lay.addWidget(self.le)

        # enter button
        self.enter_btn = QtGui.QPushButton("Enter")
        self.container_lay.addWidget(self.enter_btn)
        self.enter_btn.clicked.connect(self.run) # No '()' on run as referencing the method not calling it

        # display
        self.container_lay.addWidget(QtGui.QLabel("Accuracy:"))
        self.ans = QtGui.QLabel()
        self.container_lay.addWidget(self.ans)

	# time display
	self.container_lay.addWidget(QtGui.QLabel("Time:"))
        self.timer = QtGui.QLabel()
        self.container_lay.addWidget(self.timer)

	# output graph
	self.graph_btn = QtGui.QPushButton("Plot output")
        self.container_lay.addWidget(self.graph_btn)
        self.graph_btn.clicked.connect(self.plot) 

    def get_fname1(self):
	fname = QtGui.QFileDialog.getOpenFileName(self, 'Select file')
        if fname:
            self.original=fname
        else:
	    print "file not selected"

    def get_fname2(self):
	fname = QtGui.QFileDialog.getOpenFileName(self, 'Select file')
        if fname:
            self.ground=fname
        else:
	    print "file not selected"

    def KNN_classifier(self,k,original,ground):
	print(sys.argv)         # prints filename as it is the first argument given to console
	starttime=time.time()

	ds =gdal.Open(str(original))
	inimage = ds.ReadAsArray()
	grnd = gdal.Open(str(ground))
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

	train_cols=int(0.8*new_cols)
	test_cols=new_cols-train_cols
	train_ds=mat[:,:train_cols]         # partioning dataset into testing and training data based on a random ratio
	test_ds=mat[:,train_cols:]
	self.out= np.zeros((rows,cols))

	for i in range(test_cols):
		mat_dist=np.zeros((train_cols,2))
		for j in range(train_cols):
			dist=np.sqrt(np.sum((train_ds[:new_rows,j]-test_ds[:new_rows,i])**2))
			mat_dist[j,0]=dist			        # storing distances
			mat_dist[j,1]=mat[bands,j]		        # storing class numbers
		new_dist=mat_dist[np.argsort(mat_dist[:,0])]                         # to sort by distances                                     
		counter= np.zeros((17,))
		for x in range(int(k)):
			counter[int(new_dist[x][1])]+=1
		class_index = np.argmax(counter)                        # check case of np.count_nonzero(counter == max(counter))>1:
		self.out[int(test_ds[bands+1,i]),int(test_ds[bands+2,i])] =int(class_index)  

	for i in range(0,rows):
	    for j in range(0,cols):
		if ground[i,j]==0:
		    self.out[i,j]=0
		    
	for i in range(train_cols):
		self.out[int(train_ds[bands+1,i]),int(train_ds[bands+2,i])]=int(train_ds[bands,i])  # class numbers of train data acc original indices
			                                                                       
	c=0
	for i in range(rows):
		for j in range(cols):
			if self.out[i,j]==ground[i,j]:
		    		c+=1
		    

	Accuracy = ((c-train_cols)*100.0)/test_cols                  # as training data shouldnt be used for accuracy calculations
	endtime=time.time()
	self.Time=endtime-starttime	
	return Accuracy

    def run(self):
        Kvalue = self.le.text()
        accuracy= str(self.KNN_classifier(Kvalue,self.original,self.ground))
        self.ans.setText(accuracy)
	self.timer.setText(str(self.Time))

    def plot(self):
	from matplotlib import pyplot as plt
        plt.imshow(self.out, interpolation='nearest')
        plt.show()
	 

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)       # creates an application first then window
    window = Window()
    window.show()
    sys.exit(app.exec_())


