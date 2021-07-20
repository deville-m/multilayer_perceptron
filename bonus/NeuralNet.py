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
    maxi = np.max(x, axis=1).reshape(-1, 1)
    shiftx = x - maxi
    exps = np.exp(shiftx)
    A = exps / np.sum(exps, axis=1, keepdims=True)
    return A

class Dense:
    def __init__(self, units, input_shape, activation=sigmoid):
        self.units = units
        self.input_shape = input_shape
        self.activation = activation
        self.isout = False
        self.W = np.random.randn(input_shape, units)
        self.b = np.random.randn(1, units)

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

    def gradient_descent(self, A, rate):
        """Apply gradient descent to update the weights

        Args:
            A: previous layer's output
        Returns:
            Current layer's output
        """
        m = A.shape[0]
        reg = 0.0

        #calculate gradient
        nabla_W = (A.T @ self.delta) / m
        nabla_b = np.sum(self.delta, axis=0, keepdims=True) / m

        #update weights and biases
        self.W -= rate * nabla_W + reg * np.sum(self.W) / m
        self.b -= rate * nabla_b + reg * np.sum(self.b) / m

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
        L = (- (y.T @ np.log(h)) - ((1 - y).T @ np.log(1 - h))).diagonal()
        return L.mean() / m

    def forward(self, X, save_output=False):
        for layer in self.layers:
            X = layer.forward(X, save_output)
        return X

    def backward(self, X, y, rate):

        ## Backpropagate the error
        delta, W = self.layers[-1].backward(y)
        # Here delta and W belong to the L+1 layer
        for i in range(self.depth - 2, -1, -1):
            delta, W = self.layers[i].backward(delta, W)

        ## Gradient descent
        A = X
        for i in range(0, self.depth):
            A = self.layers[i].gradient_descent(A, rate)

    def train(self, ds, lab, epoch=100, rate=0.1, batch=None, verbose=False, valid=None):
        if verbose:
            print(f"Training with epoch: {epoch}, rate: {rate}, batch: {batch}")
        if batch is None:
            batch = len(ds)
        batch = np.ceil(len(ds) / batch)
        loss = []
        val_loss = []
        for i in range(epoch):
            idx = np.random.permutation(ds.shape[0])
            ds, lab = ds[idx], lab[idx]
            for X, y in zip(np.array_split(ds, batch), np.array_split(lab, batch)):
                self.forward(X, save_output=True)
                self.backward(X, y, rate)
            loss.append(self.loss(ds, lab))
            if valid is not None:
                val_loss.append(self.loss(valid[0], valid[1]))
            if verbose:
                if valid is None:
                    print(f"Epoch: {i + 1} / {epoch} -- loss: {loss[-1]:.5f}")
                else:
                    print(f"Epoch: {i + 1} / {epoch} -- loss: {loss[-1]:.5f} -- val_loss: {val_loss[-1]:.5f}")
        return np.array(loss).ravel(), np.array(val_loss).ravel()

    def save(self, fname):
        np.save(fname, [(layer.W, layer.b, layer.activation) for layer in self.layers])

    def load(self, fname):
        if len(fname) >= 4 and fname[-4:] != ".npy":
            fname += ".npy"
        print(f"Loading {fname}")
        try:
            data = np.load(fname, allow_pickle=True)
        except Exception as e:
            print("Cannot load weights:", e, file=sys.stderr)
            sys.exit(1)

        for W, b, activation in data:
            layer = Dense(W.shape[1], W.shape[0], activation=activation)
            layer.W = W
            layer.b = b
            self.append(layer)
    
    def random_init(self, sigma=1, mu=0, seed=None):
        if seed:
            np.random.seed(seed)
        for layer in self.layers:
            layer.W = np.random.randn(layer.input_shape, layer.units) * sigma + mu
            layer.b = np.random.randn(1, layer.units)
    
    def grid_search(self, X, y, valid, epoch=[100], rate=[0.1], batch=[None]):
        mini = 1000
        res = {}
        for e in epoch:
            for r in rate:
                for b in batch:
                    self.random_init(seed=10000)
                    _, val = self.train(X, y, e, r, b, verbose=False, valid=valid)
                    print(f"e: {e}, r: {r}, b: {b}, loss: {val[-1]}")
                    if val[-1] < mini:
                        mini = val[-1]
                        res['epoch'] = e
                        res['rate'] = r
                        res['batch'] = b
                        res['loss'] = val[-1]
        return res

def PCA(X, n_components):
    K = X.shape[0]

    cov = X.T @ X / K
    eigvals, eigvecs = np.linalg.eig(cov)

    order = np.argsort(eigvals)[::-1]
    components = eigvecs[:, order[:n_components]]

    return components, eigvals[order]