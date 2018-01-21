# Tic-Tac-Toe classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset import
dataset = pd.read_csv('tic-tac-toe.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Categorical Data
from sklearn.preprocessing import LabelEncoder
label_x = LabelEncoder()
X[:,0] = label_x.fit_transform(X[:,0])
for i in range(1,9):
    X[:,i] = label_x.transform(X[:,i])    
pass

label_y = LabelEncoder()
y = label_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, train_size = 0.8, random_state = 0)

# Regressor Decision Tree
from sklearn.tree import DecisionTreeClassifier
regressor = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
regressor.fit(X_train, y_train)

# Prediction
y_pred = regressor.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true = y_test, y_pred = y_pred)

# New Data
x_new = ['x','b','o','o','x','b','x','o','x']
y_new = ['positive']
x_new = label_x.transform(x_new)
y_new = label_y.transform(y_new)

# New Predict
y_pred_2 = regressor.predict(x_new)

# Write
row = 'x,b,o,o,x,b,x,o,x,positive'

f = open('tic-tac-toe.csv','a')
f.write(row)
f.close()