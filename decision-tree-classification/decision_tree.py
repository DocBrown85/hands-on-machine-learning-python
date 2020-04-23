# Decision Tree

# Importing libraries
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv("CustomerList.csv")
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values

# Split dataset into train and test set
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)

# Fitting Decision Tree to Training set
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)

# Predicting Test set results
Y_pred = classifier.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)
