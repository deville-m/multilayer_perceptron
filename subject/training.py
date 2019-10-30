#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from NeuralNet import NeuralNet, Dense, softmax
from sklearn.metrics import accuracy_score

def fetch(f):
    #parse training data
    df = pd.read_csv(f)
    labels = df.pop("Diagnostic")
    X, y = df.values.astype(float), labels.values.astype(int)
    y = y.reshape(-1, 1)
    y = np.array([[0, 1] if x else [1, 0] for x in y])
    return X, y

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", type=int, default=1000, help="number of epochs")
    parser.add_argument("-r", "--rate", type=float, default=0.1, help="learning rate")
    parser.add_argument("-l", "--layers", type=int, default=3, help="number of layers")
    parser.add_argument("-b", "--batch", type=int, default=None, help="set batch size")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable verbose mode")
    parser.add_argument("-g", "--graph", action="store_true", help="show learning graphs")
    parser.add_argument("-s", "--save", default="weights.npy", help="save the weights to a file")
    parser.add_argument("--load", default=None, help="load weights")
    parser.add_argument("training_dataset")
    return parser

if __name__=="__main__":
    parser = init_parser()
    args = parser.parse_args()

    X, y = fetch(args.training_dataset)

    np.random.seed(2)
    #initialize the network
    NN = NeuralNet(load=args.load)

    if args.load is None:
        temp = X.shape[1]
        while args.layers >= 2:
            NN.append(Dense(10, temp, learning_rate=args.rate))
            temp = 10
            args.layers -= 1
        NN.append(Dense(2, temp, activation=softmax, learning_rate=args.rate))

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

    NN.save(args.save)