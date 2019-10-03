#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score

def sigmoid(x):
    return expit(x)

def d_sigmoid(x):
    return x * (1 - x)

class NNLayer:
    def __init__(self, s_in, s_out, isout=False, learning_rate=0.01):
        self.s_in = s_in
        self.s_out = s_out
        self.isout = isout
        self.W = np.random.randn(s_in, s_out)
        self.b = np.zeros(s_out)

    def forward(self, X):
        self.input = X
        self.output = sigmoid(X.dot(self.W) + self.b)
        self.deriv = d_sigmoid(self.output)
        return self.output

    def compute_delta(self, e, W):
        if self.isout:
            self.delta = (self.output - e) * self.deriv
            return self.delta, self.W
        self.delta = e.dot(W.T) * self.deriv
        return self.delta, self.W

class NeuralNet:
    def __init__(self, depth, s_in, s_out, learning_rate=0.01):
        self.layers = []
        self.depth = depth
        self.learning_rate = learning_rate
        decr = s_in // depth
        out = s_in
        for i in range(depth - 1):
            self.layers.append(NNLayer(s_in, out, learning_rate=learning_rate))
            s_in, out = out, s_in - decr
        self.layers.append(NNLayer(s_in, s_out, isout=True, learning_rate=learning_rate))

    def loss(self, y):
        h = self.layers[-1].output
        m = y.shape[0]
        return (- y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) / m

    def forward(self, X):
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, X, y):
        delta, W = self.layers[-1].compute_delta(y, None)
        for i in range(self.depth - 2, -1, -1):
            layer = self.layers[i]
            delta, W = layer.compute_delta(delta, W)

    def gradient_descent(self, X):
        # Update for every layer except the 1st
        for i in range(1, self.depth):
            prev = self.layers[i - 1]
            layer = self.layers[i]
            m = layer.delta.shape[0]
            layer.W -= self.learning_rate * (np.sum(prev.output.T.dot(layer.delta), axis=0) / m)
            layer.b -= self.learning_rate * (np.sum(layer.delta, axis=0) / m)

        # Update the 1st layer with X
        m = self.layers[0].delta.shape[0]
        self.layers[0].W -= self.learning_rate * (np.sum(X.T.dot(layer.delta), axis=0) / m)
        self.layers[0].b -= self.learning_rate * (np.sum(layer.delta, axis=0) / m)

    def update_weights(self, X, y):
        self.backward(X, y)
        self.gradient_descent(X)

if __name__=="__main__":
    df = pd.read_csv("data.csv", header=None, index_col=0)
    labels = df.pop(1).eq('M').mul(1)
    data = (df - df.mean()) / df.std()
    X, y = data.values.astype(float), labels.values.astype(int)
    y = y.reshape(-1, 1)

    network = NeuralNet(2, X.shape[1], 1, learning_rate=0.1)

    loss = []
    for i in range(100):
        network.forward(X)
        loss.append(network.loss(y))
        network.update_weights(X, y)
    loss = np.array(loss).ravel()
    x = np.arange(len(loss))
    plt.plot(x, loss)
    plt.show()

    y_h = np.around(network.forward(X))
    print("Accuracy on training:", accuracy_score(y, y_h))
