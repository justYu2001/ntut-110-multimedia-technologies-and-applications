# colab 也有放一份

import gzip
import numpy as np
import cv2
from sklearn import svm, metrics

capture = cv2.VideoCapture('mnist.avi')

trainData = []

while capture.isOpened():
    retval, frame = capture.read()
    
    if retval:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        trainData.append(gray)
    else:
        capture.release()

# 歸一化
trainData = np.array(trainData) / 255
trainData = trainData.reshape(60000, -1)

file = gzip.open('train-labels-idx1-ubyte.gz', 'r')
trainLabels = []
file.read(8)
for i in range(60000):
    buffer = file.read(1)
    label = np.frombuffer(buffer, dtype = np.uint8).astype(np.int64)
    trainLabels.append(label[0])

# trainLabels = []

# with open('train-trainLabels-idx1-ubyte', 'rb') as file:
#     file.read(8)
#     for i in range(60000):
#         buffer = file.read(1)
#         label = np.frombuffer(buffer, dtype = np.uint8).astype(np.int64)
#         trainLabels.append(label[0])

file = gzip.open('t10k-images-idx3-ubyte.gz', 'r')
file.read(16)
buffer = file.read(28 * 28 * 10000)
testData = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
testData = testData.reshape(10000, 28, 28, 1)
testData = testData.reshape(10000, -1)
testData = testData / 255
    
testLabels = []
file = gzip.open('t10k-labels-idx1-ubyte.gz', 'r')
file.read(8)

for i in range(10000):
    buffer = file.read(1)
    label = np.frombuffer(buffer, np.uint8).astype(np.int64)
    testLabels.append(label)

classifier = svm.SVC()
classifier.fit(trainData, trainLabels)
predicted = classifier.predict(testData)

print(
    f"Classification report for classifier:\n"
    f"{metrics.classification_report(testLabels, predicted)}\n"
)

print('Performace:')
testLabels = np.array(testLabels).reshape(10000)
print(np.sum(testLabels == predicted) / predicted.size)