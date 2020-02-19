from sklearn.linear_model import LinearRegression
import pandas as pd
from sys import argv
import numpy as np


def readCSVFile(filePath):
    return pd.read_csv(filePath, delimiter=',')


def linearRegression(x, y):
    linReg = LinearRegression(fit_intercept=True)
    linReg.fit(x, y)
    return linReg


def fitResiduals(x, yTrue, yPred):
    yRes = yTrue - yPred
    # print(yRes)
    return linearRegression(x, yRes)


def main():
    # np.random.seed(1)
    inputFile = argv[1]
    dataFromFile = readCSVFile(inputFile)
    trainX, trainY = dataFromFile['x'], dataFromFile['y']
    trainX = trainX.values.reshape(-1, 1)
    linearRegrModel = linearRegression(trainX, trainY)
    print(linearRegrModel.score(trainX, trainY))
    print('Co-Efficients:', linearRegrModel.coef_)
    print('Intercept:', linearRegrModel.intercept_)
    trainYPred = linearRegrModel.predict(trainX)
    print(trainYPred)
    meanSquaredError = np.square(np.subtract(trainY, trainYPred)).mean()
    print(meanSquaredError)
    linearRegrModelOnRes = fitResiduals(trainX, trainY, trainYPred)
    print('Co-Efficients:', linearRegrModelOnRes.coef_)
    print('Intercept:', linearRegrModelOnRes.intercept_)
    newCoeff = linearRegrModel.coef_ + linearRegrModelOnRes.coef_
    newIntercept = linearRegrModel.intercept_ + linearRegrModelOnRes.intercept_
    newY = newCoeff * trainX + newIntercept
    newY = newY.reshape(newY.shape[0] * newY.shape[1],)
    print(trainY.shape)
    print(newY)
    meanSquaredError = np.square(np.subtract(trainY, newY)).mean()
    print(meanSquaredError)
    testX = [[2], [3]]
    print(linearRegrModel.predict(testX))


if __name__ == '__main__':
    main()
