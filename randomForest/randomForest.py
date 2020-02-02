# Random Forest
#   c.alexandracorreia@ua.pt
#   c.alexandracorreua@av.it.pt

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def single_decision_tree(rseed, x, y):
    tree = DecisionTreeClassifier(random_state=rseed)
    tree.fit(x,y)
    return tree

def limit_max_depth(rseed, max_depth, x, y):
    tree = DecisionTreeClassifier(max_depth = 2, random_state=rseed)
    tree.fit(x,y)
    return tree


