#!/usr/bin/env python3
import argparse
import csv
import pandas as pd
import numpy as np
from NeuralNet import PCA
from sklearn.metrics import accuracy_score

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
    parser.add_argument("-p", "--pca", type=int, default=None, help="apply PCA to dataset")
    parser.add_argument("dataset")
    return parser

def write_csv(fname, X, y):
    with open(fname, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["Diagnostic"] + [f"feature {i}" for i in range(X.shape[1])])
        for features, label in zip(X, y):
            writer.writerow(list(label) + list(features))

if __name__=="__main__":
    parser = init_parser()
    args = parser.parse_args()
    X, y = preprocessing(args.dataset)
    if args.pca is not None:
        X, P, V = PCA(X, args.pca)
        print(f"Variance extracted: {np.sum(V[:args.pca]) / np.sum(V) * 100:.1f}%")
    
    size = X.shape[0]
    
    np.random.seed(6)
    idx = np.random.permutation(size)
    X, y = X[idx], y[idx]
    
    train = int(size * .6)
    valid = int(size * .8)
    X_train, y_train = X[:train], y[:train]
    X_valid, y_valid = X[train:valid], y[train:valid]
    X_test, y_test = X[valid:], y[valid:]
    
    write_csv("train.csv", X_train, y_train)
    write_csv("validation.csv", X_valid, y_valid)
    write_csv("test.csv", X_test, y_test)