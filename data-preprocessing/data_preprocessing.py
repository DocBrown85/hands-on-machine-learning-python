# -*- coding: utf-8 -*-
"""
"""

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Importing Dataset
dataset = pd.read_csv("Data.csv")
dataset.describe()

# Get independent and dependent variables
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Work with missing data
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding Categorical Data (independent variable)
ct = ColumnTransformer(transformers=[("one_hot_encoder", OneHotEncoder(categories="auto"),[0])],
                       remainder="passthrough")
X = np.array(ct.fit_transform(X), dtype=np.float)

# Encoding dependent variables
le = LabelEncoder()
Y = le.fit_transform(Y)

# Split Dataset into Train and Test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.20, random_state=0)

# Apply Feature Scaling
sc_X = StandardScaler()
sc_X = sc_X.fit(X)
X_train = sc_X.transform(X_train)
X_test = sc_X.transform(X_test)
