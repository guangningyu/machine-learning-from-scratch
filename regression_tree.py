#!/usr/bin/env python
# -*- coding: utf-8 -*-

import urllib2
from numpy import mat, eye, nonzero, mean, var, shape, inf, sum, power, ones, linalg, \
        corrcoef, zeros

def createDataSet(url):
    '''
    创建数据集，最后一列为目标变量
    '''
    lines = urllib2.urlopen(url).readlines()
    dataSet = [map(float, line.strip().split('\t')) for line in lines]
    return mat(dataSet)

def binSplitDataSet(dataSet, splitColIdx, thres):
    '''
    按指定的列和指定的阈值切分数据集
    >>> dataSet
    matrix([[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
    >>> mat0, mat1 = binSplitDataSet(dataSet, 1, 0.5)
    >>> mat1
    matrix([[0, 1, 0, 0]]) # the 2nd row is selected
    >>> mat0
    matrix([[1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
    '''
    mat1 = dataSet[nonzero(dataSet[:, splitColIdx] >  thres)[0], :]
    mat0 = dataSet[nonzero(dataSet[:, splitColIdx] <= thres)[0], :]
    return mat1, mat0

# ---------- 回归树 ---------- #

def regLeaf(dataSet):
    '''
    叶节点模型：回归树中，叶节点模型就是目标变量的均值
    '''
    return mean(dataSet[:, -1])

def regErr(dataSet):
    '''
    误差估计函数：返回总方差，即目标变量的均方差乘以数据集中样本的个数
    '''
    return var(dataSet[:, -1]) * shape(dataSet)[0]

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    '''
    找到数据的最佳二元切分方式
    '''
    # 设定预剪枝参数（注意：模型结果对该参数十分敏感）
    tolS = ops[0] # 误差的最少下降值
    tolN = ops[1] # 切分后的最少样本数

    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        # 如果目标变量的值都相等，则退出
        return None, leafType(dataSet)
    else:
        m, n = shape(dataSet)
        S = errType(dataSet)
        # 初始化
        bestS = inf
        bestIdx = 0
        bestValue = 0
        # 对每个特征:
        for colIdx in range(n - 1):
            # 对每个特征值:
            for value in set(dataSet[:, colIdx].T.tolist()[0]):
                # 将数据集切分成两份
                mat1, mat0 = binSplitDataSet(dataSet, colIdx, value)
                if (shape(mat1)[0] < tolN) or (shape(mat0)[0] < tolN):
                    continue
                else:
                    # 计算切分的误差
                    newS = errType(mat1) + errType(mat0)
                    # 如果切分后的误差小于当前最小误差，则将当前切分设定为最佳切分
                    if newS < bestS:
                        bestIdx = colIdx
                        bestValue = value
                        bestS = newS
        # 如果满足预剪枝的条件，则直接创建叶节点
        if (S - bestS) < tolS:
            return None, leafType(dataSet)
        else:
            mat1, mat0 = binSplitDataSet(dataSet, bestIdx, bestValue)
            if (shape(mat1)[0] < tolN) or (shape(mat0)[0] < tolN):
                return None, leafType(dataSet)
            else:
                # 返回最佳切分的特征和阈值
                return bestIdx, bestValue

def createTree(dataSet, leafType, errType, ops):
    '''
    递归地创建树
    '''
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    else:
        returnTree = {}
        returnTree['spInd'] = feat
        returnTree['spVal'] = val
        leftSet, rightSet = binSplitDataSet(dataSet, feat, val)
        returnTree['left']  = createTree(leftSet,  leafType, errType, ops)
        returnTree['right'] = createTree(rightSet, leafType, errType, ops)
        return returnTree

# ---------- 剪枝 ---------- #

def isTree(obj):
    return (type(obj).__name__ == 'dict')

def getMean(tree):
    '''
    对树进行塌陷处理，即返回树的平均值
    '''
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0

def prune(tree, testData):
    '''
    后剪枝
    '''
    # 如果测试数据集为空，则进行剪枝（意味着出现过拟合）
    if shape(testData)[0] == 0:
        return getMean(tree)
    else:
        # 如果存在任一子集是一棵树，则在该子集递归地进行剪枝
        if isTree(tree['right']) or isTree(tree['left']):
            lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
            if isTree(tree['left']):
                tree['left'] = prune(tree['left'], lSet)
            if isTree(tree['right']):
                tree['right'] = prune(tree['right'], rSet)
            return tree
        # 如果子集均不是树（即均为叶节点），则判断是否进行合并
        else:
            lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
            # 计算不合并的误差
            errorNoMerge = sum(power(lSet[:, -1] - tree['left'],  2)) \
                         + sum(power(rSet[:, -1] - tree['right'], 2))
            # 计算合并后的误差
            treeMean = (tree['left'] + tree['right']) / 2.0
            errorMerge = sum(power(testData[:, -1] - treeMean, 2))
            # 如果合并会降低误差，则合并叶节点
            if errorMerge < errorNoMerge:
                print 'merging'
                return treeMean
            else:
                return tree

# ---------- 模型树 ---------- #

def linearSolver(dataSet):
    m, n = shape(dataSet)
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse, \n\
                        try increasing the second value of ops')
    else:
        ws = xTx.I * (X.T * Y)
        return ws, X, Y

def modelLeaf(dataSet):
    ws, X, Y = linearSolver(dataSet)
    return ws

def modelErr(dataSet):
    ws, X, Y = linearSolver(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))

# ---------- 比较预测误差 ---------- #

def regTreeEval(model, dataSet):
    return float(model)

def modelTreeEval(model, dataSet):
    n = shape(dataSet)[1]
    X = mat(ones((1, n+1)))
    X[:, 1:n+1] = dataSet
    return float(X * model)

def treeForecast(tree, inData, modelEval=regTreeEval):
    '''
    inData是一条记录
    '''
    if not isTree(tree):
        return modelEval(tree, inData)
    else:
        if inData[tree['spInd']] > tree['spVal']:
            if isTree(tree['left']):
                return treeForecast(tree['left'], inData, modelEval)
            else:
                return modelEval(tree['left'], inData)
        else:
            if isTree(tree['right']):
                return treeForecast(tree['right'], inData, modelEval)
            else:
                return modelEval(tree['right'], inData)

def createForecast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForecast(tree, mat(testData[i]), modelEval)
    return yHat


if __name__ == '__main__':
    # 1.建立回归树（regression tree，每个叶节点包含单个值）
    dataSet = createDataSet('https://raw.githubusercontent.com/pbharrin/machinelearninginaction/master/Ch09/ex2.txt')
    regTree = createTree(dataSet, regLeaf, regErr, (0, 1))
    print regTree

    #   对回归树进行剪枝
    testSet = createDataSet('https://raw.githubusercontent.com/pbharrin/machinelearninginaction/master/Ch09/ex2test.txt')
    prunedRegTree = prune(regTree, testSet)
    print prunedRegTree

    # 2.建立模型树（model tree，每个叶节点包含一个线性方程）
    dataSet = createDataSet('https://raw.githubusercontent.com/pbharrin/machinelearninginaction/master/Ch09/exp2.txt')
    modelTree = createTree(dataSet, modelLeaf, modelErr, (1, 10))
    print modelTree

    # 3.比较回归树和模型树的预测误差
    dataSet = createDataSet('https://raw.githubusercontent.com/pbharrin/machinelearninginaction/master/Ch09/bikeSpeedVsIq_train.txt')
    testSet = createDataSet('https://raw.githubusercontent.com/pbharrin/machinelearninginaction/master/Ch09/bikeSpeedVsIq_test.txt')
    #   计算回归树的R方
    regTree = createTree(dataSet, regLeaf, regErr, (1, 20))
    yHat = createForecast(regTree, testSet[:, 0], regTreeEval)
    print 'R square of regTree is:   %f' % corrcoef(yHat, testSet[:, 1], rowvar=0)[0, 1]
    #   计算模型树的R方
    modelTree = createTree(dataSet, modelLeaf, modelErr, (1, 20))
    yHat = createForecast(modelTree, testSet[:, 0], modelTreeEval)
    print 'R square of modelTree is: %f' % corrcoef(yHat, testSet[:, 1], rowvar=0)[0, 1]

