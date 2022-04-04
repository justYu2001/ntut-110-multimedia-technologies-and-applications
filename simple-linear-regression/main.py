import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sympy import reshape

def getBetaZero(x, y):
    xBar = np.mean(x)
    yBar = np.mean(y)

    numerator = np.sum((x - xBar) * (y - yBar))
    
    denominator = np.sum((x - xBar) ** 2)

    return numerator / denominator

x = np.array([])
y = np.array([])

with open('data.txt', 'r') as file:
    for line in file.readlines():
        data = line.split()
        x = np.append(x, float(data[0]))
        y = np.append(y, float(data[1]))


plt.figure(1)
plt.plot(x, y, 'o', color='blue')


betaZero = getBetaZero(x, y)

xBar = np.mean(x)
yBar = np.mean(y)
betaOne = yBar - betaZero * xBar

yHead = betaZero * x + betaOne
plt.figure(2)
plt.title(f'y = {betaZero}x + {betaOne}')
plt.plot(x, y, 'o', color='blue')
plt.plot(x, yHead, color = 'red')

linearRegression = LinearRegression()
linearRegression.fit(np.reshape(x, (-1, 1)), y)
yPredicted = linearRegression.predict(np.reshape(x, (-1, 1)))

plt.figure(3)
plt.title(f'y = {linearRegression.coef_[0]}x + {linearRegression.intercept_}')
plt.plot(x, y, 'o', color='blue')
plt.plot(x, yPredicted, color = 'black')
plt.show()