# All Imports 
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from keras.datasets import mnist
import tensorflow.keras as kb
from tensorflow.keras import backend
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

from plotnine import *

from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler #Z-score variables

from sklearn.model_selection import train_test_split # simple TT split cv
from sklearn.model_selection import KFold # k-fold cv

#Commands & Code to Get UCI Data
# !pip install ucimlrepo (Typed in Command Line)

from pandas.core.arrays.sparse.accessor import SparseFrameAccessor
from ucimlrepo import fetch_ucirepo

## fetch dataset
spambase = fetch_ucirepo(id=94)

# data (as pandas dataframes)
data = spambase.data.features
spam = spambase.data.targets

data["class"] = spam #Combines the categorical column for spam into the same table as features

#Split, Z-Score Data

features = [i for i in data.columns if i != "class"]
X = data[features]
Y = data["class"]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2)

z = StandardScaler()
X_train[features] = z.fit_transform(X_train[features])
X_test[features] = z.transform(X_test[features])

# Neural Network

model = kb.Sequential([
    kb.layers.Dense(50, input_shape =[57]),#input
    kb.layers.Dense(45),
    kb.layers.Dropout(0.2),
    kb.layers.Dense(40),
    kb.layers.Dense(35),
    kb.layers.Dropout(0.2),
    kb.layers.Dense(30),
    kb.layers.Dense(25),
    kb.layers.Dropout(0.2),
    kb.layers.Dense(20),
    kb.layers.Dense(15),
    kb.layers.Dropout(0.2),
    kb.layers.Dense(10),
    kb.layers.Dense(5),
    kb.layers.Dense(1, activation = "sigmoid") #output
])

model.compile(loss="binary_crossentropy", optimizer=kb.optimizers.SGD(0.01),
	metrics=["accuracy"])

#fit the model (same as SKlearn)
model.fit(X_train,Y_train, epochs = 25, validation_data=(X_test, Y_test))

NN_train_loss, NN_train_accuracy = model.evaluate(X_train, Y_train)
NN_test_loss, NN_test_accuracy = model.evaluate(X_test, Y_test)

#Logistic Regression

lr = LogisticRegression(penalty='l2')

lr.fit(X_train, Y_train)

print("Train Accuracy: ", accuracy_score(Y_train, lr.predict(X_train)))
print("Test Accuracy: ", accuracy_score(Y_test, lr.predict(X_test)))

# PERFOROMANCE: Train - ~0.93 | Test - ~ 0.91

