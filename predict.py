import cPickle
import numpy as np
from load import make_result_csv

predict = open('predict_cnn.cpickle').read()
predict = cPickle.loads(predict)

classes = open('classes.cpickle').read()
classes = cPickle.loads(classes)

predict = predict.astype(np.int32)


index = np.argmax(predict, axis=1)

result = [classes[i] for i in index]

make_result_csv(result, 'result_cnn')