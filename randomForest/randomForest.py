# Random Forest
#   c.alexandracorreia@ua.pt
#   c.alexandracorreia@av.it.pt

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Just a simple decicion tree
def single_decision_tree(rseed, x, y):
    tree = DecisionTreeClassifier(random_state=rseed)
    tree.fit(x,y)
    return tree

# Give a maximum depth to all trees
def limit_max_depth(rseed, max_depth, x, y):
    tree = DecisionTreeClassifier(max_depth = 2, random_state=rseed)
    tree.fit(x,y)
    return tree

#TODO: verificar se rseed = n_estimators
def with_dataset(path, rseed):
    #Assign column names to the dataset
    headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
    #read dataset to pandas dataframe
    dataset = pd.read_csv(path, names = headernames)
    dataset.head()
    # .iloc[] is primarily integer position based (from 0 to length-1 of the axis), but may also be used with a boolean array
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values
    # divide the data into train and test split: split the dataset into 70% training data and 30% of testing data −
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
    # train the model
    classifier = RandomForestClassifier(n_estimators = rseed)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    result = confusion_matrix(y_test, y_pred)
    result1 = classification_report(y_test, y_pred)
    result2 = accuracy_score(y_test,y_pred)
    return result, result1, result2