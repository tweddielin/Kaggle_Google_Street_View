import numpy as np
import os
import csv
import natsort
import cv2
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt
import csv

datasets_dir = '/media/datasets/'
datadir = "/Users/Apple/Documents/Kaggle/First Step with Julia/Data/train/"
labeldir = "/Users/Apple/Documents/Kaggle/First Step with Julia/Data/trainLabels.csv"
datadir2 = "/Users/Apple/Documents/Kaggle/First Step with Julia/Data/test/"


def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h

def mnist(ntrain=60000,ntest=10000,onehot=True):
	data_dir = os.path.join(datasets_dir,'mnist/')
	fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trX = loaded[16:].reshape((60000,28*28)).astype(float)

	fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trY = loaded[8:].reshape((60000))

	fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teX = loaded[16:].reshape((10000,28*28)).astype(float)

	fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teY = loaded[8:].reshape((10000))

	trX = trX/255.
	teX = teX/255.

	trX = trX[:ntrain]
	trY = trY[:ntrain]

	teX = teX[:ntest]
	teY = teY[:ntest]

	if onehot:
		trY = one_hot(trY, 10)
		teY = one_hot(teY, 10)
	else:
		trY = np.asarray(trY)
		teY = np.asarray(teY)

	print trX.shape
	print trY.shape
	print teX.shape
	print teY.shape
	
	trX = trX.astype(np.float32)
	teX = teX.astype(np.float32)
	trY = trY.astype(np.float32)
	teY = trY.astype(np.float32)
	return trX,teX,trY,teY


def kaggle_ocr(test_size = 0.4, image_size = 32):
	
	# Label
	label = []
	with open(labeldir,'r') as l:
		next(l)
		for row in csv.reader(l):
			label.append(row)
		l.close()
	label = np.array(label)
	label = label.T[1]
	classes = np.unique(label)
	classes = classes.tolist()
	for l in range(len(label)):
		label[l] = classes.index(label[l])

	label = np.array(label, dtype =int)
	label = one_hot(label, len(np.unique(label)))
	label = label.astype(np.float32)

	# File
	files = os.listdir(datadir)
	files.remove('.DS_Store')
	files = natsort.natsorted(files)
	data = []
	for n in range(len(files)):
		im = cv2.imread(datadir+files[n], 0)
		im = cv2.resize(im, (image_size,image_size), interpolation = cv2.INTER_LINEAR)
		im = cv2.medianBlur(src=im,ksize=5)
		data.append(np.reshape(im,image_size*image_size))
	data = np.array(data).astype(np.float32)
	

	# Split into training and testing data
	trX, teX, trY, teY = train_test_split(data, label, test_size=test_size)

	trX = trX/255.
	teX = teX/255.

	

	# test for the training data and label

	print trX.shape
	print trY.shape
	print teX.shape
	print teY.shape
	#for i in range(5):
	#	print classes[np.argmax(trY[i])]
	#	plt.figure()
	#	plt.imshow(trX[i].reshape(image_size,image_size), cmap='gray')
	#	plt.axis('off')
	#	plt.show()

	return trX,teX,trY,teY,classes


def kaggle_ocr_test(image_size = 32):
	files = os.listdir(datadir2)
	files.remove('.DS_Store')
	files = natsort.natsorted(files)
	data = []
	for n in range(len(files)):
		im = cv2.imread(datadir2+files[n],0)
		im = cv2.resize(im, (image_size,image_size), interpolation = cv2.INTER_LINEAR)
		im = cv2.medianBlur(src=im,ksize=5)
		data.append(np.reshape(im,image_size*image_size))
	test_data = np.array(data).astype(np.float32)
	
	return test_data


def make_result_csv(predict, which_method):
	datadir2 = "/Users/Apple/Documents/Kaggle/First Step with Julia/Data/testResized/"
	files = os.listdir(datadir2)
	files.remove('.DS_Store')
	files = natsort.natsorted(files)
	files = [os.path.splitext(s)[0] for s in files]

	result_name = 'result_'+which_method+'.csv'
	with open(result_name,'w') as csvfile:
		fieldnames = ['ID', 'Class']
		writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
		writer.writeheader()

		for f in range(len(predict)):
			writer.writerow({'ID': files[f], 'Class':predict[f]})



if __name__ == '__main__':
	#mnist()
	kaggle_ocr()