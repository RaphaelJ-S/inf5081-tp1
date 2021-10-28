import jupyter
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import scipy
import graphviz
from sklearn import tree
from sklearn.datasets import load_iris
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
    X, y = data.sliceLabel()
    arbre = decision_tree(X, y)
    tree.plot_tree(arbre)
    plt.show()


if __name__ == "__main__":
    main()
