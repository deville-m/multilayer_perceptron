#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
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
        self.learning_rate = learning_rate
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

    def update_weights(self, prev):
        m = self.delta.shape[0]
        self.W -= self.learning_rate * (np.sum(prev.T.dot(self.delta), axis=0) / m)
        self.b -= self.learning_rate * (np.sum(self.delta, axis=0) / m)


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
            layer.update_weights(prev.output)
        # Update the 1st layer with X
        self.layers[0].update_weights(X)

    def update_weights(self, X, y):
        self.backward(X, y)
        self.gradient_descent(X)

    def train(self, ds, lab, epoch=10000, batch=None):
        if batch is None:
            batch = len(ds)
        batch = np.ceil(len(ds) / batch)
        loss = []
        for i in range(epoch):
            idx = np.random.permutation(ds.shape[0])
            ds, lab = ds[idx], lab[idx]
            for X, y in zip(np.array_split(ds, batch), np.array_split(lab, batch)):
                self.forward(X)
                loss.append(self.loss(y))
                self.update_weights(X, y)
            print("Epoch:", i, "/", epoch, "-- learning_rate:", self.learning_rate, "-- loss:", loss[-1])
        return np.array(loss).ravel()

def preprocessing(f):
    #parse and preprocess the data
    df = pd.read_csv(f, header=None, index_col=0)
    labels = df.pop(1).eq('M').mul(1)
    data = (df - df.mean()) / df.std()
    X, y = data.values.astype(float), labels.values.astype(int)
    y = y.reshape(-1, 1)
    return X, y

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", type=int, default=1000, help="number of epochs")
    parser.add_argument("-r", "--rate", type=float, default=0.01, help="learning rate")
    parser.add_argument("-l", "--layers", type=int, default=2, help="number of layers")
    parser.add_argument("-b", "--batch", type=int, default=None, help="set batch size")
    parser.add_argument("-g", "--graph", action="store_true", default=False, help="show learning graphs")
    parser.add_argument("training_dataset")
    parser.add_argument("-t", "--test", default=None)
    return parser

if __name__=="__main__":
    parser = init_parser()
    args = parser.parse_args()

    X, y = preprocessing(args.training_dataset)

    #initialize the network
    NN = NeuralNet(args.layers, X.shape[1], 1, learning_rate=args.rate)

    loss = NN.train(X, y, args.epoch, args.batch)

    if args.graph:
        #plot the loss
        x = np.arange(len(loss))
        plt.plot(x, loss)
        plt.ylabel("loss(y_h)")
        plt.xlabel("epoch")
        plt.title("MLP using logistic function")
        plt.show()

    #calc accuracy
    y_h = np.around(NN.forward(X))
    print("Accuracy on training:", accuracy_score(y, y_h))

    if args.test:
        X, y = preprocessing(args.test)

        #pass the test
        y_h = np.around(NN.forward(X))
        print("Accuracy on test:", accuracy_score(y, y_h))
