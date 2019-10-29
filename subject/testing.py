#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from NeuralNet import NeuralNet, Dense, softmax
from sklearn.metrics import accuracy_score

def fetch(f):
    #parse testing data
    df = pd.read_csv(f)
    labels = df.pop("Diagnostic")
    X, y = df.values.astype(float), labels.values.astype(int)
    y = y.reshape(-1, 1)
    y = np.array([[0, 1] if x else [1, 0] for x in y])
    return X, y

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", default=None, help="load weights")
    parser.add_argument("testing_dataset")
    return parser

if __name__=="__main__":
    parser = init_parser()
    args = parser.parse_args()

    X, y = fetch(args.testing_dataset)

    #initialize the network
    NN = NeuralNet(load=args.load)

    if args.load is None:
        NN.append(Dense(10, X.shape[1]))
        NN.append(Dense(10, 10))
        NN.append(Dense(2, 10, activation=softmax))

    y_h = np.around(NN.forward(X))
    print("Accuracy on test:", accuracy_score(y, y_h))