#!/usr/bin/env python
# -*- coding: utf-8 -*-

import urllib2
import numpy as np


def load_simple_data():
    features = ([
         [1. , 2.1]
        ,[2. , 1.1]
        ,[1.3, 1. ]
        ,[1. , 1. ]
        ,[2. , 1. ]
    ])
    labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return features, labels


def load_data(url):
    features = []
    labels   = []
    for line in urllib2.urlopen(url).readlines():
        line = [float(i) for i in line.strip().split('\t')]
        labels.append(line.pop())
        features.append(line)
    return features, labels


def stump_classify(features_mat, col_idx, thres, criteria):
    m = np.shape(features_mat)[0]
    pred_mat = np.ones((m, 1))
    # set the prediction to -1.0 if satisfies the criteria
    if criteria == 'lt':
        pred_mat[features_mat[:, col_idx] <= thres] = -1.0
    else:
        pred_mat[features_mat[:, col_idx] >  thres] = -1.0
    return pred_mat


def calculate_error_rate(pred, labels):
    m = len(labels)
    agg_err = np.multiply(pred != np.mat(labels).T, np.ones((m, 1)))
    err_rate = agg_err.sum() / m
    return err_rate


def build_stump(features, labels, D, verbose=0):
    '''
    This is a weak classifier. It will choose a cut-off value of one column to get the minimum
    weighted error.
    '''
    features_mat  = np.mat(features)
    labels_mat = np.mat(labels).T

    # m: number of samples; n: number of features
    m, n = np.shape(features_mat)

    # init params
    step_num = 10
    best_stump = {}
    best_est = np.mat(np.zeros((m, 1)))
    min_err = np.inf

    # loop through each feature
    for col_idx in range(n):
        col_min = features_mat[:, col_idx].min()
        col_max = features_mat[:, col_idx].max()
        step_size = (col_max - col_min) / step_num

        # loop through each cut value
        for j in range(-1, int(step_num)+1):
            for criteria in ['lt', 'gt']:
                # run prediction
                thres = col_min + float(j) * step_size
                pred_mat = stump_classify(features_mat, col_idx, thres, criteria)

                # calculate weighted error
                err_mat = np.mat(np.ones((m, 1)))
                err_mat[pred_mat == labels_mat] = 0
                weighted_err = D.T * err_mat
                if verbose:
                    print("split: dim %d, thres: %.2f, criteria: %s, weighted error: %.3f" % \
                        (col_idx, thres, criteria, weighted_err))

                if weighted_err < min_err:
                    min_err = weighted_err
                    best_est = pred_mat.copy()
                    best_stump['col_idx']  = col_idx
                    best_stump['thres']    = thres
                    best_stump['criteria'] = criteria

    return best_stump, min_err, best_est


def train_adaboost(features, labels, iters=40):
    adaboost_classifier = []
    m, n = np.shape(features)

    # init params
    D = np.mat(np.ones((m, 1))/m)       # records' weights
    agg_est = np.mat(np.zeros((m, 1)))  # aggregated prediction

    for i in range(iters):
        print('\n| Iter %s...' % i)

        # run the weak classifier
        best_stump, min_err, best_est = build_stump(features, labels, D)
        print('| - Input weights: %s' % D.T)
        print('| - Predict: %s' % best_est.T)

        # calculate alpha
        alpha = float(0.5 * np.log((1.0 - min_err)/max(min_err, 1e-16)))
        best_stump['alpha'] = alpha
        adaboost_classifier.append(best_stump)

        # update D
        # if pred is right, "expon" will be -alpha; otherwise "expon" will be alpha
        expon = np.multiply(-1 * alpha * np.mat(labels).T, best_est)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()

        # update aggregated prediction
        agg_est += alpha * best_est
        print('| - Aggregated prediction: %s' % agg_est.T)

        # calculate error rate
        err_rate = calculate_error_rate(np.sign(agg_est), labels)
        print('| - Total error: %s' % err_rate)
        if err_rate == 0.0:
            break

    return adaboost_classifier


def adaboost_classify(features, adaboost_classifer, verbose=0):
    if verbose:
        print('\n| Classify: %s...' % features)
    features_mat = np.mat(features)
    m, n = np.shape(features_mat)
    agg_est = np.mat(np.zeros((m, 1)))
    for i in range(len(adaboost_classifier)):
        stump = adaboost_classifier[i]
        pred = stump_classify(features_mat, stump['col_idx'], stump['thres'], stump['criteria'])
        agg_est += stump['alpha'] * pred
        if verbose:
            print('| - Prediction: %s' % agg_est.T)
    if verbose:
        print('| - Final prediction: %s' % np.sign(agg_est.T))
    return np.sign(agg_est)


if __name__ == '__main__':
    # train
    trainset = 'https://raw.github.com/pbharrin/machinelearninginaction/master/Ch07/horseColicTraining2.txt'
    features, labels = load_data(trainset)
    adaboost_classifier = train_adaboost(features, labels, iters=10)

    # classify
    testset  = 'https://raw.github.com/pbharrin/machinelearninginaction/master/Ch07/horseColicTest2.txt'
    features, labels = load_data(testset)
    pred = adaboost_classify(features, adaboost_classifier, verbose=0)
    err_rate = calculate_error_rate(pred, labels)
    print('Error rate: %s' % err_rate)
