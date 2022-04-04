import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def getBeta(x, y):
    xt = np.transpose(x)
    a = np.dot(xt, x)
    b = np.dot(xt, y)
    result = np.dot(np.linalg.inv(a), b)
    return result

weight = []
verticalLength = []
diagonalLength = []
crossLength = []
height = []
width = []

with open('data.txt', 'r') as file:
    for line in file.readlines():
        data = line.split()
        weight += [float(data[0])]
        verticalLength += [float(data[1])]
        diagonalLength += [float(data[2])]
        crossLength += [float(data[3])]
        height += [float(data[4])]
        width += [float(data[5])]

y = np.array(weight)
ones = [1 for i in range(len(weight))]
x = np.array([ones, verticalLength, diagonalLength, crossLength, height, width])
x = x.transpose()
beta = getBeta(x, y)
yHead = np.dot(x, beta.transpose())

linearRegression = LinearRegression()
linearRegression.fit(x, y)
coefficient = [linearRegression.intercept_, *linearRegression.coef_[1:]]
yPredicted = linearRegression.predict(x)

print(coefficient)
print(beta)