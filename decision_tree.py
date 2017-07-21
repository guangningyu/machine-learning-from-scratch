#!/usr/bin/env python

from math import log
import operator
import urllib2

def createDataSet():
    '''
    prepare data: the last column is the label
    dataSet is like:
    [['young', 'myope', 'no', 'reduced', 'no lenses'],
     ['young', 'myope', 'no', 'normal', 'soft'],
     ['young', 'myope', 'yes', 'reduced', 'no lenses'],
     ...
     ['young', 'myope', 'yes', 'normal', 'hard'],
     ['young', 'hyper', 'no', 'reduced', 'no lenses']]
    '''
    lines = urllib2.urlopen('https://raw.githubusercontent.com/pbharrin/machinelearninginaction/master/Ch03/lenses.txt').readlines()
    dataSet = [line.strip().split('\t') for line in lines]
    featureNames = ['age', 'prescript', 'astigmatic', 'tearRate']
    return dataSet, featureNames

def calcShannonEnt(dataSet):
    '''
    Calculate the entropy of the label column (i.e. the last column of the data set)
    '''
    # count the frequency of each label
    labelCounts = {}
    for rec in dataSet:
        label = rec[-1]
        if (label not in labelCounts.keys()):
            labelCounts[label] = 0
        labelCounts[label] += 1
    # calculate the entropy
    totalCounts = len(dataSet)
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / totalCounts
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def sliceDataSet(dataSet, axis, value):
    '''
    return the sub data set where the axis column's value == value, excluding
    the axis column.
    '''
    subDataSet = []
    for rec in dataSet:
        if (rec[axis] == value):
            subDataSet.append(rec[:axis] + rec[axis+1:])
    return subDataSet

def chooseBestFeatureToSplit(dataSet):
    recNum = len(dataSet)
    featureNum = len(dataSet[0]) - 1 # the last column is the label
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    # calculate entropy for each feature
    for i in range(featureNum):
        featureList = [rec[i] for rec in dataSet]
        uniqFeatureValues = set(featureList)
        newEntropy = 0.0
        for value in uniqFeatureValues:
            # select the sub data set where this feature's value == value
            subDataSet = sliceDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(recNum)
            newEntropy += prob * calcShannonEnt(subDataSet)
        # select the best feature: which feature makes the entropy decrease the most
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    '''
    return the class with highest frequency in the class list
    '''
    classCounts = {}
    for vote in classList:
        if (vote not in classCounts.keys()):
            classCounts[vote] = 0
        classCounts[vote] += 1
    sortedClassCounts = sorted(classCounts.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCounts[0][0]

def createTree(dataSet, featureNames):
    '''
    create the decision tree using ID3 algorithm
    '''
    labelList = [rec[-1] for rec in dataSet]
    if (len(set(labelList)) == 1):
        # if the labels are all the same, return the label's value
        return labelList[0]
    elif (len(dataSet[0]) == 1):
        # if only lable column is left, return the label's value with highest frequency
        return majorityCnt(labelList)
    else:
        # select best feature
        bestFeatureIndex = chooseBestFeatureToSplit(dataSet)
        bestFeatureName = featureNames[bestFeatureIndex]
        # init tree
        myTree = {bestFeatureName: {}}
        del(featureNames[bestFeatureIndex])
        # get the uniq values of the best feature
        bestFeatureValues = [rec[bestFeatureIndex] for rec in dataSet]
        uniqBestFeatureValues = set(bestFeatureValues)
        # for each uniq value, generate the tree recursively
        for value in uniqBestFeatureValues:
            subFeatureNames = featureNames[:] # create a copy of featureNames
            myTree[bestFeatureName][value] = createTree(sliceDataSet(dataSet, bestFeatureIndex, value), subFeatureNames)
        return myTree

def classify(tree, featureNames, testRec):
    rootFeatureName = tree.keys()[0]
    rootFeatureDict = tree[rootFeatureName]
    rootFeatureIndex = featureNames.index(rootFeatureName)
    for rootFeatureValue in rootFeatureDict.keys():
        if (testRec[rootFeatureIndex] == rootFeatureValue):
            nextBranch = rootFeatureDict[rootFeatureValue]
            if (type(nextBranch).__name__ == 'dict'):
                # if this branch's value is still a dict, classify recursively
                featureName = classify(nextBranch, featureNames, testRec)
            else:
                featureName = nextBranch
    return featureName


if __name__ == '__main__':
    dataSet, featureNames = createDataSet()
    tree = createTree(dataSet, featureNames[:])
    featureName = classify(tree, featureNames, ['young', 'hyper', 'yes', 'normal'])
