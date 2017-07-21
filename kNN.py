#!/usr/bin/env python

from numpy import array, tile, zeros, shape
import operator
import urllib2
import matplotlib
import matplotlib.pyplot as plt

def createDataSet():
    '''
    dataSet is like:
    [[  4.09200000e+04   8.32697600e+00   9.53952000e-01]
     [  1.44880000e+04   7.15346900e+00   1.67390400e+00]
     [  2.60520000e+04   1.44187100e+00   8.05124000e-01]
     ...,
     [  2.65750000e+04   1.06501020e+01   8.66627000e-01]
     [  4.81110000e+04   9.13452800e+00   7.28045000e-01]
     [  4.37570000e+04   7.88260100e+00   1.33244600e+00]]

    labels is like:
    [3, 2, 1, ..., 2, 3, 1]
    '''
    lines = urllib2.urlopen('https://raw.githubusercontent.com/pbharrin/machinelearninginaction/master/Ch02/datingTestSet2.txt').readlines()
    dataSet = zeros((len(lines), 3))
    labels = []
    index = 0
    for line in lines:
        line = line.strip().split('\t')
        dataSet[index, :] = line[0:3]
        labels.append(int(line[-1]))
        index += 1
    return dataSet, labels

def plotDataSet(dataSet, labels):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1) # 1*1 grid, 1st subplot
    ax.scatter(dataSet[:, 1], dataSet[:, 2],
        15.0*array(labels), 15.0*array(labels))
    plt.show()

def autoNorm(dataSet):
    '''
    Normalize dataSet: normVlu = (vlu - min) / (max - min)
    '''
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def classify0(inX, dataSet, labels, k):
    '''
    Classify inX using kNN
    '''
    # calculate Euclidean distance
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    # sort
    sortedDistIndicies = distances.argsort()
    # count label frequency among top k data points
    classCount = {}
    for i in range(k):
        label = labels[sortedDistIndicies[i]]
        classCount[label] = classCount.get(label, 0) + 1
    # select the label with highest frequency
    sortedClassCount = sorted(classCount.iteritems(),
        key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def runClassification(dataSet, labels, testRatio):
    # normalization
    dataSet, ranges, minVals = autoNorm(dataSet)
    # run classification
    dataSetSize = dataSet.shape[0]
    testSetSize = int(dataSetSize*testRatio)
    trainDataSet = dataSet[testSetSize:dataSetSize, :]
    trainLabels = labels[testSetSize:dataSetSize]
    errorNum = 0
    for i in range(testSetSize):
        testDataPoint = dataSet[i, :]
        k = 3
        testLabel = classify0(testDataPoint, trainDataSet, trainLabels, k)
        realLabel = labels[i]
        print("The classifier came back with: %d, the real answer is: %d" % (testLabel, realLabel))
        if (testLabel != realLabel):
            errorNum += 1
    print("The totle error rate is: %f" % (float(errorNum)/float(testSetSize)))

if __name__ == '__main__':
    # prepare data
    dataSet, labels = createDataSet()
    # plot data
    plotDataSet(dataSet, labels)
    # run classification
    runClassification(dataSet, labels, testRatio=0.1)
