# Data Pre-processing

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Importing dataset
dataset = pd.read_csv("OrganizationData.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding independent variable
indexOfTheColumnToEncode = 3
ct = ColumnTransformer(transformers=[('one_hot_encoder',OneHotEncoder(categories='auto'),[indexOfTheColumnToEncode])],
                                     remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# Avoid Dummy Variable Trap (exclude one dummy variable)
X = X[:, 1:]

# Split dataset into train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.20,random_state=0)

# Fitting Multiple Linear Regression to Training Set
regressor = LinearRegression()

regressor.fit(X_train, Y_train)

# Predicting Test Set Result
Y_pred = regressor.predict(X_test)

# Optimizing Model using Backward Elimination
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)

X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()