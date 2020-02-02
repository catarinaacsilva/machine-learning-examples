import numpy as np
import pandas as pd
from randomForest import single_decision_tree, limit_max_depth


#set random seed to ensure reproductible runs
rseed = 50

x = np.array([[2, 2], [2, 1], [2, 3], [1, 2], [1, 1], [3, 3]])
y = np.array([0, 1, 1, 1, 0, 1])

tree = single_decision_tree(rseed, x, y)
print(f'Decision tree has {tree.tree_.node_count} nodes with maximum depth {tree.tree_.max_depth}.')
print(f'Model Accuracy: {tree.score(x, y)}')

tree = limit_max_depth(rseed, 2, x, y)
print(f'Decision tree has {tree.tree_.node_count} nodes with maximum depth {tree.tree_.max_depth}.')
print(f'Model Accuracy: {tree.score(x, y)}')