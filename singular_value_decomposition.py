#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Calculate the singular value decomposition using the power method.
'''

import numpy as np
from numpy.linalg import norm
from random import normalvariate
from math import sqrt


def random_unit_vector(n):
    '''
    return a random unit vector of n-dimension
    '''
    unnormalized = [normalvariate(0, 1) for i in range(n)]
    denom = sqrt(sum(x*x for x in unnormalized))
    return [x / denom for x in unnormalized]


def svd_1d(A, epsilon=1e-10):
    '''
    one-dimensional SVD, return the unit singular vector
    '''
    m, n = A.shape
    x = random_unit_vector(n)
    B = np.dot(A.T, A)

    last_value = None
    this_value = x
    iterations = 0
    while True:
        iterations += 1
        last_value = this_value
        this_value = np.dot(B, this_value)
        # normalize this_value to a unit vector
        this_value = this_value / norm(this_value)

        if abs(np.dot(this_value, last_value)) > (1 - epsilon):
            print('converged in {} iterations.'.format(iterations))
            return this_value


def svd(A, epsilon=1e-10):
    m, n = A.shape
    svd_so_far = []

    for i in range(n):
        matrix_for_1d = A.copy()

        for u, sigma, v in svd_so_far:
            matrix_for_1d -= sigma * np.outer(u, v)

        v = svd_1d(matrix_for_1d, epsilon=epsilon)
        u_unnormalized = np.dot(A, v)
        sigma = norm(u_unnormalized)
        u = u_unnormalized / sigma

        svd_so_far.append((u, sigma, v))

    # transform into the right shapes
    u, sigma, v = [np.array(x) for x in zip(*svd_so_far)]

    return u.T, sigma, v



if __name__ == '__main__':
    movie_ratings = np.array([
        [2, 5, 3],
        [1, 2, 1],
        [4, 1, 1],
        [3, 5, 2],
        [5, 3, 1],
        [4, 5, 5],
        [2, 4, 2],
        [2, 2, 5],
    ], dtype='float64')

    u, sigma, v = svd(movie_ratings)
    print(u)
    print(sigma)
    print(v)

    err = np.round(movie_ratings - np.dot(u, np.dot(np.diag(sigma), v)), decimals=10)
    print(err)
