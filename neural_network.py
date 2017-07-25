#!/usr/bin/env python

import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt


def load_data(n_samples=100, noise=None):
    np.random.seed(0)
    return sklearn.datasets.make_moons(n_samples=n_samples, noise=noise)


def init_network_weights(network, seed=0):
    '''
    initialize the weights of the network
    network[0]: number of input nodes
    network[1]: number of hidden nodes
    network[2]: number of output nodes
    '''
    np.random.seed(seed)
    W1 = np.random.randn(network[0], network[1]) / np.sqrt(network[0])
    b1 = np.zeros((1, network[1]))
    W2 = np.random.rand(network[1], network[2]) / np.sqrt(network[1])
    b2 = np.zeros((1, network[2]))
    return W1, b1, W2, b2


def forward_propagation(X, W1, b1, W2, b2):
    '''
    make predictions using forward propagation:
    zi is the input of layer i
    ai is the output of layer i after applying the activation function
    '''
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    # softmax
    exp_scores = np.exp(z2)
    a2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return z1, a1, z2, a2


def backpropagation(X, Y, a1, a2, W2):
    '''
    calculate gradients using backpropagation
    '''
    # encode target
    Y = [[0, 1] if i == 1 else [1, 0] for i in Y]
    delta3 = a2 - Y
    dW2 = (a1.T).dot(delta3)
    db2 = np.sum(delta3, axis=0, keepdims=True)
    delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
    dW1 = np.dot(X.T, delta2)
    db1 = np.sum(delta2, axis=0)
    return dW1, db1, dW2, db2


def add_regularization(reg_lambda, W1, b1, W2, b2, dW1, db1, dW2, db2):
    dW2 += reg_lambda * W2
    dW1 += reg_lambda * W1
    return dW1, db1, dW2, db2


def update_weights(epsilon, W1, b1, W2, b2, dW1, db1, dW2, db2):
    W1 += -epsilon * dW1
    b1 += -epsilon * db1
    W2 += -epsilon * dW2
    b2 += -epsilon * db2
    return W1, b1, W2, b2


def calculate_loss(X, Y, reg_lambda, W1, b1, W2, b2):
    '''
    evaluate the total loss on the dataset
    '''
    # calculate predictions
    outputs = (z1, a1, z2, a2) = forward_propagation(X, W1, b1, W2, b2)
    # calculate loss
    correct_probs = a2[range(len(X)), Y] # if y=1, prob = a2[1]; else prob = a2[0]
    data_loss = np.sum(-np.log(correct_probs))
    # add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./len(X) * data_loss


def run_nn(X, Y, network, epsilon=0.01, reg_lambda=0.01, iters=100, seed=10, verbose=False):
    # init the network's weights
    weights = (W1, b1, W2, b2) = init_network_weights(network, seed=seed)

    for i in range(iters):
        # forward propagation
        outputs = (z1, a1, z2, a2) = forward_propagation(X, *weights)
        # backpropagation
        grad = (dW1, db1, dW2, db2) = backpropagation(X, Y, a1, a2, W2)
        # add regularization terms
        grad = (dW1, db1, dW2, db2) = add_regularization(reg_lambda, *(weights+grad))
        # update weights
        weights = (W1, b1, W2, b2) = update_weights(epsilon, *(weights+grad))
        # print loss for each step
        if verbose and i % 1000 == 0:
            print('Loss after iteration %i: %f' % (i, calculate_loss(X, Y, reg_lambda, *weights)))

    return weights


def predict(X, W1, b1, W2, b2):
    outputs = (z1, a1, z2, a2) = forward_propagation(X, *weights)
    return np.argmax(a2, axis=1)


def plot_decision_boundary(X, Y, pred_func):
    # set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)


if __name__ == '__main__':
    # load data
    X, Y = load_data(200, 0.20)
    #plt.scatter(X[:, 0], X[:, 1], s=40, c=Y, cmap=plt.cm.Spectral)
    #plt.show()

    # init model params
    num_input  = 2
    num_hidden = 3
    num_output = 2
    params = {
        'network':     (num_input, num_hidden, num_output)
        ,'epsilon':    0.01  # learning rate
        ,'reg_lambda': 0.01  # regularization strength
        ,'iters':      20000
        ,'seed':       0
        ,'verbose':    True
    }

    # run neural network
    weights = run_nn(X, Y, **params)

    # plot the decision boundary
    plot_decision_boundary(X, Y, lambda x: predict(x, *weights))
    plt.title('Decision Boundary for hidden layer size %d' % num_hidden)
    plt.show()
