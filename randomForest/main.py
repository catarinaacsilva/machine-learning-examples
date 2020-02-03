import numpy as np
import pandas as pd
from randomForest import single_decision_tree, limit_max_depth


def no_maximumDepth(rseed, x, y):
    tree = single_decision_tree(rseed, x, y)
    print(f'Decision tree has {tree.tree_.node_count} nodes without maximum depth {tree.tree_.max_depth}.')
    print(f'Model Accuracy: {tree.score(x, y)}')

def maximunDepth(rseed, max_depth, x, y):
    tree = limit_max_depth(rseed, max_depth, x, y)
    print(f'Decision tree has {tree.tree_.node_count} nodes with maximum depth {tree.tree_.max_depth}.')
    print(f'Model Accuracy: {tree.score(x, y)}')

def with_dataset(path):
    data = pd.read_cvs(path).sample(100000, random_state = RSEED)
    df.head()


def main():
    #set random seed to ensure reproductible runs
    rseed = 50
    x = np.array([[2, 2], [2, 1], [2, 3], [1, 2], [1, 1], [3, 3]])
    y = np.array([0, 1, 1, 1, 0, 1])
    max_depth = 2

    no_maximumDepth(rseed, x, y)
    maximunDepth(rseed, max_depth, x, y)

main()
