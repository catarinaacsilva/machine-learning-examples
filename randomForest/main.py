import numpy as np
import pandas as pd
from randomForest import single_decision_tree, limit_max_depth, with_dataset



def no_maximumDepth(rseed, x, y):
    tree = single_decision_tree(rseed, x, y)
    print(f'Decision tree has {tree.tree_.node_count} nodes without maximum depth {tree.tree_.max_depth}.')
    print(f'Model Accuracy: {tree.score(x, y)}')

def maximunDepth(rseed, max_depth, x, y):
    tree = limit_max_depth(rseed, max_depth, x, y)
    print(f'Decision tree has {tree.tree_.node_count} nodes with maximum depth {tree.tree_.max_depth}.')
    print(f'Model Accuracy: {tree.score(x, y)}')

def readDataset(path, rseed):
    result, result1, result2 = with_dataset(path, rseed)
    print("Confusion Matrix:")
    print(result)
    print("Classification Report:",)
    print (result1)
    print("Accuracy:",result2)


    

def main():
    #set random seed to ensure reproductible runs
    rseed = 50
    x = np.array([[2, 2], [2, 1], [2, 3], [1, 2], [1, 1], [3, 3]])
    y = np.array([0, 1, 1, 1, 0, 1])
    max_depth = 2
    no_maximumDepth(rseed, x, y)
    maximunDepth(rseed, max_depth, x, y)
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    readDataset(path, rseed)

main()
