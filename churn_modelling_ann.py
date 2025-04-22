# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 10:15:02 2025

@author: abhir
"""

# Artificial Neural Network

# Part 1 - Data Preprocessing


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading the dataset
dataset = pd.read_csv("C:/Users/abhir/Jupyter notebook/Deep Learning/ANN/churn modelling/Churn_Modelling.csv")


# Creating X and y variable
X = dataset.iloc[:,3:13]
y = dataset.iloc[:,13]


# Creating dummy veriables
geography = pd.get_dummies(X['Geography'], drop_first=True)
gender = pd.get_dummies(X['Gender'], drop_first=True)


# Concatinate the DataFrames
X = pd.concat([X,geography,gender], axis=1)


# Dropping unnessary column
X = X.drop(['Geography','Gender'], axis=1)


# Splitting the dataset in Train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)


# Part 2 - Now let's make ANN

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout


# Initialize the ANN
classifier = Sequential()


# Adding the input layer and the first hidden  layer

classifier.add(Dense(units = 10, kernel_initializer = 'he_uniform', activation='relu', input_dim = 11))
### input_dim  = 11 -> because input features are 11 in the X dataframe
### Dense layer -> to create first hidden layer
### units = 6 -> 6 neurons in the first hidden layer
### kernel_initializer -> How the weights need to initialize
### activation -> Using ReLU activation function


classifier.add(Dropout(0.3))

# Adding the second hidden layer
classifier.add(Dense(units = 20, kernel_initializer = 'he_uniform',activation='relu'))

classifier.add(Dropout(0.4))

# Adding the third hidden layer
classifier.add(Dense(units = 15, kernel_initializer = 'he_uniform',activation='relu'))

classifier.add(Dropout(0.2))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation='sigmoid'))


# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fitting the ANN to the training set
model_history = classifier.fit(X_train, y_train, validation_split = 0.33, batch_size = 10, epochs=100)


# Predicting the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)




# List all data in the history
print(model_history.history.keys())

# Summerize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.legend(['train','test'], loc = 'upper left')
plt.show()



# Summerize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.legend(['train','test'], loc = 'upper left')
plt.show()
