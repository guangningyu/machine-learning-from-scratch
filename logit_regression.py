#!/usr/bin/env python
# -*- coding: utf-8 -*-

import urllib2
from numpy import mat, ones, shape, exp, array, arange
import matplotlib.pyplot as plt

def createDataSet():
    features = []
    labels = []
    lines = urllib2.urlopen('https://raw.github.com/pbharrin/machinelearninginaction/master/Ch05/testSet.txt').readlines()
    for line in lines:
        line = line.strip().split()
        features.append([1.0, float(line[0]), float(line[1])]) # set x0 to 1.0
        labels.append(int(line[2]))
    return features, labels

def sigmoid(value):
    return 1.0 / (1 + exp(-value))

def gradAscent(features, labels, alpha=0.001, iterations=500):
    '''
    梯度上升算法：
    - 批处理算法：每次更新回归系数时都需要遍历整个数据集
    '''
    featureMatrix = mat(features)
    labelMatrix = mat(labels).transpose()
    m, n = shape(featureMatrix)
    weights = ones((n, 1))
    for k in range(iterations):
        h = sigmoid(featureMatrix*weights)
        error = (labelMatrix - h)
        weights += alpha * featureMatrix.transpose() * error # adjust weights
    return weights

def stocGradAscent(features, labels, alpha=0.001):
    '''
    随机梯度上升算法：
    - 在线学习算法：一次仅用一个样本点来更新回归系数，可以在新样本到来时对分类器进行增量式更新
    - 与梯度上升算法相比，占用更少的计算资源
    '''
    featureMatrix = mat(features)
    m, n = shape(featureMatrix)
    weights = ones((n, 1))
    for i in range(m):
        h = sigmoid(sum(featureMatrix[i]*weights)) # h is a value
        error = labels[i] - h # error is a value
        weights += alpha * featureMatrix[i].transpose() * error
    return weights

def plotBestFit(features, labels, weights):
    recNum = shape(features)[0]
    xcord0 = []
    ycord0 = []
    xcord1 = []
    ycord1 = []
    for i in range(recNum):
        feature1 = features[i][1]
        feature2 = features[i][2]
        if (int(labels[i]) == 0):
            xcord0.append(feature1)
            ycord0.append(feature2)
        else:
            xcord1.append(feature1)
            ycord1.append(feature2)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # plot data points
    ax.scatter(xcord0, ycord0, s=30, c='red', marker='s')
    ax.scatter(xcord1, ycord1, s=30, c='green')
    # plot regression
    x = arange(-3.0, 3.0, 0.1)
    # Note:
    # Because:
    #     (1) x axis: x1, y axis: x2
    #     (2) 0 = w0*x0 + w1*x1 + w2*x2 (0 corresponds to 0.5 in sigmoid function)
    #     (3) x0 = 1.0
    # We have:
    #     x2 = -(w0 + w1*x1) / w2
    # i.e. y = -(w0 + w1*x) / w2
    y = -(weights[0] + weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.show()


if __name__ == '__main__':
    features, labels = createDataSet()
    # method 1: gradient ascent
    weights = gradAscent(features, labels)
    plotBestFit(features, labels, weights)
    # method 2: stochastic gradient ascent
    weights = stocGradAscent(features, labels, 0.01)
    plotBestFit(features, labels, weights)
