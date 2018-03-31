'''==========================================================
THIS IS A CODE FOR AN ARTIFICIAL NEURAL NETWORK IN PYTHON.
============================================================'''
#------- Importing the libraries ------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ----------- Importing the dataset --------------------
dataset = pd.read_csv('Churn_Modelling.csv');
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#--------------- Encoding categorical data ------
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
x[:, 1] = labelencoder_X_1.fit_transform(x[:, 1])

labelencoder_X_2 = LabelEncoder()
x[:, 2] = labelencoder_X_1.fit_transform(x[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:,1:]

#------- Splitting the dataset into Training and Test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#--------- Feature Scaling ------------------------------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train =sc.fit_transform(x_train)
x_test =sc.fit_transform(x_test)

#----- Fitting Classifier to Training set ------------------
#----- Create your classifier here--------------------------



#-----Predicting the Test set Results ---------------	
y_pred = classifier.predict(x_test)

#--------- Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


 

