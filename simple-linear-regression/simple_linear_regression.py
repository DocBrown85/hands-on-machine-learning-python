# -*- coding: utf-8 -*-
"""
Building a Simple Linear Regression Model
"""

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing Dataset
dataset = pd.read_csv("Salaries.csv")

# Get independent and dependent variables
X = dataset.iloc[:, 0].values
Y = dataset.iloc[:, 1].values

# Split Dataset into Train and Test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=1/3, random_state=0)

# Fitting Simple Linear Regression to training set
regressor = LinearRegression()
## X and Y has to be 2D arrays
X_train = X_train.reshape(-1,1)
Y_train = Y_train.reshape(-1,1)

regressor.fit(X_train, Y_train)

# Predicting Test Set Result
X_test = X_test.reshape(-1,1)
Y_pred = regressor.predict(X_test)

# Visualizing Result (Training Set)
plt.scatter(X_train, Y_train)
plt.plot(X_train, regressor.predict(X_train), color="red")
plt.title("Experience VS Salary (Training Set Results)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Visualizing Result (Test Set)
plt.scatter(X_test, Y_test)
plt.plot(X_train, regressor.predict(X_train), color="red")
plt.title("Experience VS Salary (Test Set Results)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()