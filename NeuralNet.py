#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.special import expit
from sklearn.metrics import accuracy_score

def relu(x, derivative=False):
    temp = np.copy(x)
    temp[temp < 0] = 0
    if derivative:
        temp[temp >= 0] = 1
    return temp

def sigmoid(x, derivative=False):
    temp = expit(x)
    if not derivative:
        return temp
    return temp * (1 - temp)

def softmax(x, derivative=False):
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    A = exps / np.sum(exps, axis=1, keepdims=True)
    return A

class Dense:
    def __init__(self, units, input_shape, activation=sigmoid, learning_rate=0.01):
        self.units = units
        self.input_shape = input_shape
        self.activation = activation
        self.isout = False
        self.learning_rate = learning_rate
        self.W = np.random.randn(input_shape, units)
        self.b = np.zeros((1, units))

    def forward(self, X, save_output=False):
        """Apply forward propagation for one layer

        Args:
            X: input from the previous layer
            save_output: if True save the intermidiate and final output
        """
        Z = X @ self.W + self.b
        A = self.activation(Z)
        if save_output:
            self.Z = Z
            self.A = A
        return A

    def backward(self, E, W=None):
        """Backpropagate the error, and set self.delta

        Args:
            E: if self.isout -> truth
               else -> next_layer.delta
            W: if self.isout -> None
               else -> next_layer.W
        Returns:
            the delta (error) for this layer
            the weights of the current layer
        """
        if self.isout:
            self.delta = self.A - E
            return self.delta, self.W

        self.deriv = self.activation(self.Z, derivative=True)
        self.delta = (E @ W.T) * self.deriv
        return self.delta, self.W

    def gradient_descent(self, A):
        """Apply gradient descent to update the weights

        Args:
            A: previous layer's output
        Returns:
            Current layer's output
        """
        m = A.shape[0]

        #calculate gradient
        nabla_W = (A.T @ self.delta) / m
        nabla_b = np.sum(self.delta, axis=0, keepdims=True) / m

        #update weights and biases
        self.W -= self.learning_rate * nabla_W
        self.b -= self.learning_rate * nabla_b

        return self.A


class NeuralNet:
    def __init__(self, load=None):
        self.layers = []
        self.depth = 0

        if load:
            self.load(load)

    def append(self, layer):
        if self.layers:
            self.layers[-1].isout = False
        layer.isout = True
        self.layers.append(layer)
        self.depth += 1

    def loss(self, X, y):
        h = self.forward(X)
        m = h.shape[0]
        return (- y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))).diagonal().sum() / m

    def forward(self, X, save_output=False):
        for layer in self.layers:
            X = layer.forward(X, save_output)
        return X

    def backward(self, X, y):

        ## Backpropagate the error
        delta, W = self.layers[-1].backward(y)
        # Here delta and W belong to the L+1 layer
        for i in range(self.depth - 2, -1, -1):
            delta, W = self.layers[i].backward(delta, W)

        ## Gradient descent
        A = X
        for i in range(0, self.depth):
            A = self.layers[i].gradient_descent(A)

    def train(self, ds, lab, epoch=100, batch=None, verbose=False):
        if batch is None:
            batch = len(ds)
        batch = np.ceil(len(ds) / batch)
        loss = []
        for i in range(epoch):
            idx = np.random.permutation(ds.shape[0])
            ds, lab = ds[idx], lab[idx]
            for X, y in zip(np.array_split(ds, batch), np.array_split(lab, batch)):
                self.forward(X, save_output=True)
                self.backward(X, y)
            loss.append(self.loss(ds, lab))
            if verbose:
                print("Epoch:", i + 1, "/", epoch, "-- learning_rate:", self.layers[-1].learning_rate, "-- loss:", loss[-1])
        return np.array(loss).ravel()

    def save(self, fname):
        np.save(fname, [(layer.W, layer.b, layer.activation, layer.learning_rate) for layer in self.layers])

    def load(self, fname):
        if len(fname) >= 4 and fname[-4:] != ".npy":
            fname += ".npy"
        print(f"Loading {fname}")
        try:
            data = np.load(fname, allow_pickle=True)
        except Exception as e:
            print("Cannot load weights:", e, file=sys.stderr)
            sys.exit(1)

        for W, b, activation, rate in data:
            layer = Dense(W.shape[1], W.shape[0], activation=activation, learning_rate=rate)
            layer.W = W
            layer.b = b
            self.append(layer)


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
    parser.add_argument("-v", "--verbose", action="store_true", help="enable verbose mode")
    parser.add_argument("-g", "--graph", action="store_true", help="show learning graphs")
    parser.add_argument("-s", "--save", help="save the weights to a file")
    parser.add_argument("--load", default=None, help="load weights")
    parser.add_argument("-t", "--test", default=None)
    parser.add_argument("training_dataset")
    return parser

if __name__=="__main__":
    parser = init_parser()
    args = parser.parse_args()

    X, y = preprocessing(args.training_dataset)

    #initialize the network
    NN = NeuralNet(load=args.load)

    if args.load is None:
        if args.layers >= 2:
            NN.append(Dense(10, X.shape[1], learning_rate=args.rate))
            for i in range(1, args.layers - 1):
                NN.append(Dense(10, 10, learning_rate=args.rate))
            NN.append(Dense(1, 10, learning_rate=args.rate))
        elif args.layers == 1:
            NN.append(Dense(1, X.shape[1], learning_rate=args.rate))

    loss = NN.train(X, y, epoch=args.epoch, batch=args.batch, verbose=args.verbose)

    if args.graph:
        #plot the loss
        plt.plot(loss)
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

    if args.save:
        NN.save(args.save)
