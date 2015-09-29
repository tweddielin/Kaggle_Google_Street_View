import numpy as np
np.random.seed(1337)  # for reproducibility

from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import cPickle
from keras.optimizers import SGD
#from load import kaggle_ocr, make_result_csv, kaggle_ocr_test


batch_size = 128
nb_classes = 62
nb_epoch = 600

# shape of the image (SHAPE x SHAPE)
shapex, shapey = 32, 32
# number of convolutional filters to use
#nb_filters = 32
# level of pooling to perform (POOL x POOL)
#nb_pool = 2
# level of convolution to perform (CONV x CONV)
#nb_conv = 3


#
#trX,teX,trY,teY,classes = kaggle_ocr(test_size = 0, image_size = 32)
kaggle_data = open('data.cpickle').read()
kaggle_data = cPickle.loads(kaggle_data)
data = kaggle_data['data']
label = kaggle_data['label']

trX, teX, trY, teY = train_test_split(data, label, test_size =0) 

trX = trX/255.
teX = teX/255.

print trX.shape[0],'x', trX.shape[1], 'train samples'
print teX.shape[0],'x', teX.shape[1], 'test samples' 

trX = trX.reshape(-1, 1, shapex, shapey)
teX = teX.reshape(-1, 1, shapex, shapey)
print trX.shape
print teX.shape
#

model = Sequential()

model.add(Convolution2D(32, 1, 3, 3, border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize = (2, 2)))
model.add(Dropout(0.1))

model.add(Convolution2D(64 ,32, 3, 3, border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize = (2,2))) 
model.add(Dropout(0.2))

model.add(Convolution2D(128,64, 3, 3, border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(128 ,128,3 ,3 ))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize = (2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
# the resulting image after conv and pooling is the original shape
# divided by the pooling with a number of filters for each "pixel"
# (the number of filters is determined by the last Conv2D)
model.add(Dense(128 * 4 * 4, 512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(512, nb_classes))
model.add(Activation('softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.7, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
early_stopping = EarlyStopping(monitor='loss', patience =2)
model.fit(trX, trY, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(teX, teY))
#score = model.evaluate(teX, teY, show_accuracy=True, verbose=0)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])

#f = open('../Models/cnn.cpickle', 'w')
#f.write(cPickle.dumps(model))
#f.close()

#model.save_weights('../Models/cnn_epoch')


#test_data = kaggle_ocr_test(image_size = 32)
test_data = open('test_data.cpickle').read()
test_data = cPickle.loads(test_data)
test_data = test_data/255.
test_data = test_data.reshape(-1, 1, shapex, shapey)
prediction = model.predict(test_data, batch_size, verbose=1)
prediction = prediction.astype(np.float32)

f = open('predict_cnn.cpickle', 'w')
f.write(cPickle.dumps(prediction))
f.close()

#make_result_csv(result, 'cnn')


