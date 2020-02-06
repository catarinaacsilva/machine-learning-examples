import numpy as np
import pandas as pd
from randomForest import single_decision_tree, limit_max_depth
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def no_maximumDepth(rseed, x, y):
    tree = single_decision_tree(rseed, x, y)
    print(f'Decision tree has {tree.tree_.node_count} nodes without maximum depth {tree.tree_.max_depth}.')
    print(f'Model Accuracy: {tree.score(x, y)}')

def maximunDepth(rseed, max_depth, x, y):
    tree = limit_max_depth(rseed, max_depth, x, y)
    print(f'Decision tree has {tree.tree_.node_count} nodes with maximum depth {tree.tree_.max_depth}.')
    print(f'Model Accuracy: {tree.score(x, y)}')

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
    

def main():
    #set random seed to ensure reproductible runs
    rseed = 50
    x = np.array([[2, 2], [2, 1], [2, 3], [1, 2], [1, 1], [3, 3]])
    y = np.array([0, 1, 1, 1, 0, 1])
    max_depth = 2
    no_maximumDepth(rseed, x, y)
    maximunDepth(rseed, max_depth, x, y)
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    result, result1, result2 =  with_dataset(path, rseed)
    print("Confusion Matrix:")
    print(result)
    print("Classification Report:",)
    print (result1)
    print("Accuracy:",result2)

main()
