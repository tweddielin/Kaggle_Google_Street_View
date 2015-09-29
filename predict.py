import cPickle
import numpy as np
import csv

test_file_name = open('test_file_name.cpickle').read()
test_file_name = cPickle.loads(test_file_name)

predict = open('predict_cnn.cpickle').read()
predict = cPickle.loads(predict)

classes = open('classes.cpickle').read()
classes = cPickle.loads(classes)
print classes
#predict = predict.astype(np.int32)


index = np.argmax(predict, axis=1)

result = [classes[i] for i in index]
print result[:10]

with open('result_cnn.csv','w') as csvfile:
	fieldnames = ['ID', 'Class']
	writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
	writer.writeheader()

	for f in range(len(result)):
		writer.writerow({'ID': test_file_name[f], 'Class': result[f]})


