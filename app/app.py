import jupyter
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import scipy
from sklearn import tree
from classes.data import Data


def load_data(file):
    data = []
    with open(file) as f:
        for line in f:
            data.append(line.strip("\n").split(","))
    return data


def decision_tree(data, classLabel):
    arbre = tree.DecisionTreeClassifier()
    return arbre.fit(data, classLabel)


def main():
    data = Data(load_data("glass.data"))
    frame = data.toDataFrame()
    frame.hist(bins=50, figsize=(20, 15))
    plt.show()
    frame[frame["Type of glass"] == 1].hist(bins=50, figsize=(20, 15))
    plt.show()


if __name__ == "__main__":
    main()
