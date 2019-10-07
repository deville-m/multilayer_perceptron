#!/usr/bin/env python3
from NeuralNet import NeuralNet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

def print_data(x):
    k = "-oiOI"
    for i in range(28):
        for j in range(28):
            idx = int(np.floor(x[i * 28 + j] * 4))
            print(k[idx], end="")
        print()

print("Fetch mnist database..")
ds, lab = fetch_openml("mnist_784", return_X_y=True)
print("Done")

ds = ds / 255
lab = np.array([[1 if i == int(x) else 0 for i in range(10)] for x in lab])

print("DB dimensions:", ds.shape)
print("Splitting database into training and testing")

X, y, X_test, y_test = ds[0:60000, :], lab[0:60000, :], ds[60000:, :], lab[60000:, :]

print("Begin training")
print("Data dimensions:", X.shape)
print("Label dimensions:", y.shape)

NN = NeuralNet(depth=2, width=10, learning_rate=0.01, s_in=784, s_out=10)

NN.train(X, y, epoch=4, batch=10, verbose=True)


for xt, yt in zip(X_test, y_test):
    print_data(xt)
    h = NN.forward(xt)
    print("Guess:", h)
    print("Solution:", yt)
    input("Press enter to continue")
