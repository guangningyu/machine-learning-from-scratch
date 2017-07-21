#!/usr/bin/env python

import re
from numpy import ones, log

def createDataSet():
    docs = [
        ['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'],
    ]
    labels = [0, 1, 0, 1, 0, 1]
    return docs, labels

def createVocabList(docs):
    '''
    create a full list of the words in docs
    '''
    vocabList = [word for doc in docs for word in doc]
    return list(set(vocabList))

def doc2Vec(vocabList, doc, model='set'):
    '''
    map vocabList to a vector:
    [set-of-words model] if the word is in the doc, then set it to 1; else set it to 0.
    [bag-of-words model] set the value to the total frequency in the doc
    '''
    docVec = [0] * len(vocabList)
    if (model == 'set'):
        for word in doc:
            if (word in vocabList):
                docVec[vocabList.index(word)] = 1
    else:
        for word in doc:
            if (word in vocabList):
                docVec[vocabList.index(word)] += 1
    return docVec

def docs2Matrix(vocabList, docs):
    docsMatrix = []
    for doc in docs:
        docsMatrix.append(doc2Vec(vocabList, doc))
    return docsMatrix

def trainNB(vocabList, docs, labels):
    '''
    train naive bayes model
    '''
    docs = docs2Matrix(vocabList, docs) # convert data set to 0/1 matrix
    docsNum = len(docs)
    wordsNum = len(docs[0])
    pPositive = sum(labels)/float(len(labels)) # positive rate
    p1Num = ones(wordsNum) # freq of each word when positive; set 1 as default value for smoothing
    p1Denom = 2.0          # freq of all the words when positive; set 2.0 as default value for smoothing
    p0Num = ones(wordsNum) # freq of each word when negative; set 1 as default value for smoothing
    p0Denom = 2.0          # freq of all the words when negative; set 2.0 as default value for smoothing
    for i in range(docsNum):
        if (labels[i] == 1):
            # if label is positive
            p1Num += docs[i]
            p1Denom += sum(docs[i])
        else:
            # if label is negative
            p0Num += docs[i]
            p0Denom += sum(docs[i])
    p1Vec = log(p1Num / p1Denom)
    p0Vec = log(p0Num / p0Denom)
    return p0Vec, p1Vec, pPositive

def classifyNB(docVec, p0Vec, p1Vec, pPositive):
    p0 = sum(docVec * p0Vec) + log(1.0 - pPositive)
    p1 = sum(docVec * p1Vec) + log(pPositive)
    return 1 if p1 > p0 else 0

def testNB(vocabList, p0Vec, p1Vec, pPositive, testDoc):
    testVec = doc2Vec(vocabList, testDoc)
    testLabel = classifyNB(testVec, p0Vec, p1Vec, pPositive)
    print testDoc, 'classified as: ', testLabel


if __name__ == '__main__':
    # prepare train data set
    docs, labels = createDataSet()
    # generate a full word list
    vocabList = createVocabList(docs)
    # train naive bayes model
    p0Vec, p1Vec, pPositive = trainNB(vocabList, docs, labels)
    # run test
    testNB(vocabList, p0Vec, p1Vec, pPositive, ['love', 'my', 'dalmation'])
    testNB(vocabList, p0Vec, p1Vec, pPositive, ['stupid', 'garbage'])
