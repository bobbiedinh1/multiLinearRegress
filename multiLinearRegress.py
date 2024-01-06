import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# X[:, :-1] = imputer.fit_transform(X[:, :-1])
# print(X)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

#---Do not have to do in multiple linear regress--
# from sklearn.preprocessing import StandardScaler
# scalar = StandardScaler()
# X_train[:, 3:] = scalar.fit_transform(X_train[:, 3:])
# X_test[:, 3:] = scalar.transform(X_test[:, 3:])

#---LIBRARY TAKES CARE OF DUMMY TRAP AND P VALUE SIGNIFICANCE---

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

#set only 2 decimal places
np.set_printoptions(precision=2)
# print(regressor.predict([[1, 0, 0, 19854, 35899, 65348]]))

#y_pred = number of rows, 1 = number of columns, axis = 1 is horizontal and =0 would be vertical
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))

#y_pred and y_test results are close in proximity
# here we can say the multiple linear regression is well adapted with the dataset

# print(np.concatenate((y_pred.reshape(1, len(y_pred)), y_test.reshape(1, len(y_test))), axis=0))