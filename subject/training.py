#!/usr/bin/env python3
import argparse
import os.path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from NeuralNet import NeuralNet, Dense, softmax, PCA
from sklearn.metrics import accuracy_score

def fetch(f, pca=None):
    #parse and preprocess the data
    if not os.path.exists(f):
        print("File does not exists")
        sys.exit(1)
    df = pd.read_csv(f, header=None, index_col=0)
    labels = df.pop(1).eq('M').mul(1)
    mean = df.mean()
    std = df.std()
    data = (df - mean) / std
    X, y = data.values.astype(float), labels.values.astype(int)
    y = y.reshape(-1, 1)
    y = np.array([[0, 1] if x else [1, 0] for x in y])
    if pca:
        P, V = PCA(X, pca)
        print(f"Variance extracted: {np.sum(V[:pca]) / np.sum(V) * 100:.2f}")
        X = X @ P
        np.save("pca.npy", P)
    np.save("normalization.npy", (mean, std), allow_pickle=True)
    return X, y

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", type=int, default=1000, help="number of epochs")
    parser.add_argument("-r", "--rate", type=float, default=0.1, help="learning rate")
    parser.add_argument("-l", "--layers", type=int, default=4, help="number of layers")
    parser.add_argument("-b", "--batch", type=int, default=None, help="set batch size")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable verbose mode")
    parser.add_argument("-g", "--graph", action="store_true", help="show learning graphs")
    parser.add_argument("-s", "--save", default="weights.npy", help="save the weights to a file")
    parser.add_argument("--load", default=None, help="load weights")
    parser.add_argument("--pca", type=int, default=None, help="apply pca to dataset")
    parser.add_argument("dataset")
    return parser

if __name__=="__main__":
    parser = init_parser()
    args = parser.parse_args()

    X, y = fetch(args.dataset, args.pca)
    split = int(len(X) * 0.8)
    valid = (X[split:], y[split:])
    X, y = X[:split], y[:split]

    #initialize the network
    NN = NeuralNet(load=args.load)

    if args.load is None:
        temp = X.shape[1]
        mid = (temp + 2) // 2
        while args.layers >= 2:
            NN.append(Dense(mid, temp))
            temp = mid
            args.layers -= 1
        NN.append(Dense(2, temp, activation=softmax))

    loss, val_loss = NN.train(X, y, epoch=args.epoch, rate=args.rate, batch=args.batch, verbose=args.verbose, valid=valid)

    if args.graph:
        #plot the loss
        plt.plot(loss, label="training")
        plt.plot(val_loss, label="cross-validation")
        plt.ylabel("loss(y_h)")
        plt.xlabel("epoch")
        plt.title("MLP using logistic function")
        plt.legend()
        plt.show()

    y_h = np.around(NN.forward(X))
    print("Accuracy on training:", accuracy_score(y, y_h))
    print("Loss on training:", NN.loss(X, y))

    y_h = np.around(NN.forward(valid[0]))
    print("Accuracy on validation:", accuracy_score(valid[1], y_h))
    print("Loss on validation:", NN.loss(valid[0], valid[1]))
    
    NN.save(args.save)