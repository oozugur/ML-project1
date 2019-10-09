#!/usr/bin/env python3
# -*- coding: utf-8 -*-

 # Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Preparing the dataset
dataset = pd.read_csv('Coastal Data System - Waves (Mooloolaba) 01-2017 to 06 - 2019.csv')
dataset = dataset.replace(-99.9,np.nan)
dataset = dataset.interpolate(limit_direction='both')
x = dataset.iloc[:,2: ].values
y = dataset.iloc[:, 1].values

#Box plot of raw data
dataset.plot(kind='box', subplots=True, layout=(5,5), sharex=False, sharey=False)
plt.show()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train , y_train)
model = regressor.fit(x_train , y_train)


#Predicting the Test-Set Results
y_pred = regressor.predict(x_test)

# Visualising results
plt.scatter(y_test, y_pred)
plt.xlabel('Real_Values')
plt.ylabel('Predictions')

#Alternative Plot 
import seaborn as sns
ax1 = sns.distplot(y_test, hist=False, color="r", label="real_val")
sns.distplot(y_pred , hist=False, color="b", label="pred_val" , ax=ax1)

#Printing score
print ('Score:', model.score(x_test, y_test))

#Performing Cross Validation 
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
scores = cross_val_score(model, x, y, cv=5)

#Printing out CrossVal scores
print ('Cross-validated scores:', scores)



 

