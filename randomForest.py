# Random Forest
#   c.alexandracorreia@ua.pt
#   c.alexandracorreua@av.it.pt

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#set random seed to ensure reproductible runs
rseed = 50

x = np.array([[2, 2], [2, 1], [2, 3], [1, 2], [1, 1], [3, 3]])
y = np.array([0, 1, 1, 1, 0, 1])

tree = DecisionTreeClassifier(random_state=rseed)
tree.fit(x,y)


print(f'Decision tree has {tree.tree_.node_count} nodes with maximum depth {tree.tree_.max_depth}.')
print(f'Model Accuracy: {tree.score(x, y)}')


tree_max_depth = DecisionTreeClassifier(max_depth = 2, random_state=rseed)
tree_max_depth.fit(x,y)


print(f'Decision tree has {tree_max_depth.tree_.node_count} nodes with maximum depth {tree_max_depth.tree_.max_depth}.')
print(f'Model Accuracy: {tree_max_depth.score(x, y)}')