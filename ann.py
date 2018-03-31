'''==========================================================
THIS IS A CODE FOR AN ARTIFICIAL NEURAL NETWORK IN PYTHON.
============================================================'''
#------- Importing the libraries ------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ----------- Importing the dataset --------------------
dataset = pd.read_conv('Churn_Modelling.csv');
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

#------- Splitting the dataset into Training and Test datasets
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 

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


 

