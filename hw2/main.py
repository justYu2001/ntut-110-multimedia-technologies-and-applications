import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

x = []

with open('iris_x.txt', 'r') as file:
    for line in file.readlines():
        data = [float(i) for i in line.split()]
        x.append(data)

y = []

with open('iris_y.txt', 'r') as file:
    for line in file.readlines():
        data = float(line)
        y.append(data)

xTrain, xTest, yTrain, yTest = train_test_split(x, y, random_state = 20220413)

logisticRegression = LogisticRegression()
logisticRegression.fit(xTrain, yTrain)
predicted = logisticRegression.predict(xTest)
mse = np.mean((predicted - yTest) ** 2)
print('MSE: {:.6}\n'.format(mse))

class QDA:
    def fit(self, X, y):
        self.priors = {}
        self.means = {}
        self.covs = {}
        self.classes = np.unique(y)

        X = np.array(X)
        y = np.array(y)

        for c in self.classes:
            Xc = X[y == c]
            self.priors[c] = Xc.shape[0] / X.shape[0]
            self.means[c] = np.mean(Xc, axis = 0)
            self.covs[c] = np.cov(Xc, rowvar = False)

    def predict(self, X):
        preds = []
        for xi in X:
            posts = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                invCov = np.linalg.inv(self.covs[c])
                invCovDet = np.linalg.det(invCov)
                diff = xi - self.means[c]
                likelihood = -0.5 * np.log(invCovDet) - 0.5 * diff.T @ invCov @ diff
                post = prior + likelihood
                posts.append(post)
            pred = self.classes[np.argmax(posts)]
            preds.append(pred)
        return np.array(preds)

qda = QDA()
qda.fit(xTrain, yTrain)
predicted = qda.predict(xTest)
confusionMatrix = confusion_matrix(yTest, predicted)
print(confusionMatrix)
acc = np.diag(confusionMatrix).sum() / len(yTest)
print(f'acc:{acc}\n')

qda = QuadraticDiscriminantAnalysis()
qda.fit(xTrain, yTrain)
predicted = qda.predict(xTest)
confusionMatrix = confusion_matrix(yTest, predicted)
print(confusionMatrix)
acc = np.diag(confusionMatrix).sum() / len(yTest)
print(f'acc:{acc}')