#!/usr/bin/env python3
import argparse
import os.path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from NeuralNet import NeuralNet, Dense, softmax
from sklearn.metrics import accuracy_score

def fetch(f):
    #parse and preprocess the data
    if not os.path.exists(f):
        print("File does not exists")
        sys.exit(1)
    df = pd.read_csv(f, header=None, index_col=0)
    if os.path.exists("normalization.npy"):
        mean, std = np.load("normalization.npy", allow_pickle=True)
    else:
        mean, std = 0, 1
    labels = df.pop(1).eq('M').mul(1)
    data = (df - mean) / std
    X, y = data.values.astype(float), labels.values.astype(int)
    y = y.reshape(-1, 1)
    y = np.array([[0, 1] if x else [1, 0] for x in y])
    if os.path.exists("pca.npy"):
        P = np.load("pca.npy")
        X = X @ P
    return X, y

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", default="weights.npy", help="load weights")
    parser.add_argument("testing_dataset")
    return parser

if __name__=="__main__":
    parser = init_parser()
    args = parser.parse_args()

    X, y = fetch(args.testing_dataset)

    if not os.path.exists(args.load):
        NN.append(Dense(10, X.shape[1]))
        NN.append(Dense(10, 10))
        NN.append(Dense(2, 10, activation=softmax))
    else:
        NN = NeuralNet(load=args.load)

    y_h = np.around(NN.forward(X))
    print("Accuracy on test:", accuracy_score(y, y_h))
    print("Loss on test:", NN.loss(X, y))