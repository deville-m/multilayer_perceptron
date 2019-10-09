#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.special import expit
from sklearn.metrics import accuracy_score

def sigmoid(x):
    return expit(x)

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NNLayer:
    def __init__(self, s_in, s_out, isout=False, learning_rate=0.01):
        self.s_in = s_in
        self.s_out = s_out
        self.isout = isout
        self.learning_rate = learning_rate
        self.W = np.random.randn(s_in, s_out)
        self.b = np.zeros((1, s_out))

    def forward(self, X, save_output=False):
        """Apply forward propagation for one layer

        Args:
            X: input from the previous layer
            save_output: if True save the intermidiate and final output
        """
        Z = X @ self.W + self.b
        A = sigmoid(Z)
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
        self.deriv = d_sigmoid(self.Z)
        if self.isout:
            self.delta = self.A - E
            return self.delta, self.W

        #deriv.shape  = (1, nb_classes)
        #E.shape      = (1, nb_classes)
        #self.W.shape = (nb_features, nb_classes)
        #self.delta = np.empty((E.shape[0], self.s_out))
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
    def __init__(self, s_in, s_out, depth, hidden=None, learning_rate=0.01, load=None):
        self.layers = []
        self.depth = depth
        self.learning_rate = learning_rate

        #initialize the layers
        if depth >= 2:
            if hidden is None:
                hidden = (s_in + s_out) // 2
            self.layers.append(NNLayer(s_in, hidden, learning_rate=learning_rate))
            for i in range(1, depth - 1):
                self.layers.append(NNLayer(hidden, hidden, learning_rate=learning_rate))
            self.layers.append(NNLayer(hidden, s_out, isout=True, learning_rate=learning_rate))
        else:
            self.layers.append(NNLayer(s_in, s_out, isout=True, learning_rate=learning_rate))

        if load:
            self.load(load)

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
                print("Epoch:", i + 1, "/", epoch, "-- learning_rate:", self.learning_rate, "-- loss:", loss[-1])
        return np.array(loss).ravel()

    def save(self, fname):
        np.save(fname, [(layer.W, layer.b) for layer in self.layers])

    def load(self, fname):
        if len(fname) >= 4  and fname[-4: -1] != ".npy":
            fname += ".npy"
        try:
            data = np.load(fname, allow_pickle=True)
            i = 0
            for d in data:
                layer = self.layers[i]
                layer.W, layer.b = d
                i += 1
        except Exception as e:
            print("Cannot load weights:", e, file=sys.stderr)
            sys.exit(1)


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
    parser.add_argument("-l", "--layers", type=int, default=3, help="number of layers")
    parser.add_argument("-b", "--batch", type=int, default=None, help="set batch size")
    parser.add_argument("-v", "--verbose", action="store_true", default=None, help="set batch size")
    parser.add_argument("-g", "--graph", action="store_true", default=False, help="show learning graphs")
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
    NN = NeuralNet(X.shape[1], 1, depth=2, learning_rate=args.rate, load=args.load)

    loss = NN.train(X, y, args.epoch, args.batch, args.verbose)

    if args.graph:
        #plot the loss
        x = np.arange(len(loss))
        plt.plot(x, loss)
        plt.ylabel("loss(y_h)")
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
