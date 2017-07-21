#!/usr/bin/env python
# -*- coding: utf-8 -*-

import urllib2
import copy
from random import seed
from random import randrange
from math import sqrt

'''
Random Forest Algorithm on Sonar Dataset

Reference: [How to Implement Random Forest From Scratch in Python](http://machinelearningmastery.com/implement-random-forest-scratch-python/)
'''

def load_data():
    '''
    get Sonar data set and preprocess it
    '''
    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'
    rows = []
    for line in urllib2.urlopen(data_url).readlines():
        line = line.strip().split(',')
        features = [float(i) for i in line[:-1]]
        label = [1 if line[-1] == 'M' else 0]
        rows.append(features + label)
    return rows


def subsample(dataset, ratio):
    '''
    simple random sampling with replacement
    '''
    sample = []
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


def test_split(dataset, index, value):
    '''
    split a dataset based on an attribute and an attribute value
    '''
    left  = []
    right = []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def gini_index(groups, class_values):
    '''
    calculate the Gini index for a split dataset
    '''
    gini = 0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += (proportion * (1.0 - proportion))
    return gini


def get_split(dataset, n_features):
    '''
    select the best split point for a dataset
    '''
    # randomly selected n features
    features = []
    while len(features) < n_features:
        index = randrange(len(dataset[0])-1)
        if index not in features:
            features.append(index)

    # get set of uniq labels
    class_values = list(set(row[-1] for row in dataset))

    # init params
    b_score  = 9999  # minimum Gini index
    b_index  = None  # index of best column
    b_value  = None  # best cut-off value of the best column
    b_groups = None  # best groups

    # loop through selected features to get the minimum Gini index
    for col_index in features:
        for row in dataset:
            col_value = row[col_index]
            groups = test_split(dataset, col_index, col_value)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index  = col_index
                b_value  = col_value
                b_score  = gini
                b_groups = groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}


def to_terminal(group):
    '''
    return the label with highest frequency
    '''
    labels = [row[-1] for row in group]
    return max(set(labels), key=labels.count)


def split(node, max_depth, min_size, n_features, depth):
    '''
    create child splits for a node or make terminal
    '''
    left, right = node['groups']
    del(node['groups'])

    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return

    # check for max depth
    if depth >= max_depth:
        node['left']  = to_terminal(left)
        node['right'] = to_terminal(right)
        return

    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth+1)

    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth+1)


def build_tree(train, max_depth, min_size, n_features):
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root


def predict(node, row):
    '''
    make a prediction with a decision tree
    '''
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def bagging_predict(trees, row):
    '''
    make a prediction with a list of bagged trees
    '''
    predictions = [predict(tree, row) for tree in trees]
    # return the label that is voted by most of the trees
    return max(set(predictions), key=predictions.count)


def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    # train
    trees = []
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    # predict
    predictions = [bagging_predict(trees, row) for row in test]
    return(predictions)


def cross_validation_split(dataset, n_folds):
    '''
    split the dataset into n folds
    '''
    dataset_copy = copy.copy(dataset)
    fold_size = int(len(dataset_copy)/n_folds)
    folds = []
    for i in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        folds.append(fold)
    return folds


def accuracy_metric(actual, predicted):
    '''
    calculate accuracy percentage
    '''
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def evaluate_algorithm(dataset, algorithm, params, n_folds=5):
    folds = cross_validation_split(dataset, n_folds)
    scores = []
    for fold in folds:
        # prepare train set
        trainset = copy.copy(folds)
        trainset.remove(fold)
        trainset = sum(trainset, []) # flatten trainset
        # prepare test set
        testset = copy.copy(fold)
        # train & predict
        predicted = algorithm(trainset, testset, **params)
        # evaluate
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


if __name__ == '__main__':
    seed(1)
    dataset = load_data()
    n_features = int(sqrt(len(dataset[0])-1))
    params = {
        'n_trees': 10                # number of trees to be built
        ,'sample_size': 1.0          # fraction to be randomly sampled for each tree
        ,'n_features':  n_features   # number of features to be randomly selected to evaluate for each split
        ,'max_depth': 10             # maximum depth of each tree
        ,'min_size': 1               # minimum records number after split
    }
    for n_trees in [1, 5, 10, 20]:
        params['n_trees'] = n_trees
        scores = evaluate_algorithm(dataset, random_forest, params, n_folds=5)
        print('Trees: %d' % n_trees)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
