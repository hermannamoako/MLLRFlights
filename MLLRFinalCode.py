# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 13:03:23 2018

@author: hkkam
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#IMPORT THE DATASET THE THREE DATASET
df1 = pd.read_csv('C:/Users/hkkam/Desktop/MLP 2.0/June.csv')
df2 = pd.read_csv('C:/Users/hkkam/Desktop/MLP 2.0/July.csv')
df3 = pd.read_csv('C:/Users/hkkam/Desktop/MLP 2.0/August.csv')

#Join the 3  dataframes  together
frames = [df1, df2, df3]
dfx = pd.concat(frames)

#Check for Some Vital flight Summaries
(dfx["CANCELLED"]).value_counts()
(dfx["DAY_OF_WEEK"]).value_counts()

#Extract ONLY flights which were succesful.   i.e Flights which were not cancelled
dfx = dfx[dfx["CANCELLED"] == 0] #A successful flight has "CANCELLED" = 0
dfx = dfx.reset_index(drop=True)

#Visualize to check the correlation between the features
sns.heatmap(dfx.corr())

#Extract the rows based on specific origin city name
dfx = dfx.loc[(dfx['ORIGIN_CITY_NAME'] == 'Atlanta, GA') | (dfx['ORIGIN_CITY_NAME'] 
== 'Chicago, IL') | (dfx['ORIGIN_CITY_NAME'] == 'New York, NY') | (dfx['ORIGIN_CITY_NAME'] 
== 'Los Angeles, CA') | (dfx['ORIGIN_CITY_NAME'] == 'Dallas, TX') ]

#Extract the rows based on specific destination city name
dfx = dfx.loc[(dfx['DEST_CITY_NAME'] == 'Atlanta, GA') | (dfx['DEST_CITY_NAME'] 
== 'Chicago, IL') | (dfx['DEST_CITY_NAME'] == 'New York, NY') | (dfx['DEST_CITY_NAME'] 
== 'Los Angeles, CA') | (dfx['DEST_CITY_NAME'] == 'Dallas, TX') ]

#Extract the rows based on the top 10 airlines in the USA
dfx = dfx.loc[(dfx['UNIQUE_CARRIER'] == 'AS') | (dfx['UNIQUE_CARRIER'] == 'DL')
 | (dfx['UNIQUE_CARRIER'] == 'VX') | (dfx['UNIQUE_CARRIER'] == 'B6') | (dfx['UNIQUE_CARRIER']
 == 'HA') | (dfx['UNIQUE_CARRIER'] == 'WN') | (dfx['UNIQUE_CARRIER'] == 'OO') |
 (dfx['UNIQUE_CARRIER'] == 'UA') | (dfx['UNIQUE_CARRIER'] == 'AA') | (dfx['UNIQUE_CARRIER'] == 'EV')]

#Reset the columns of the data frame
dfx = dfx.reset_index(drop=True)

#Code the 'ARR_DELAY' column. Early arrival = 0 and late = 1
dfx['ARR_DELAY'] = [1 if x > 0 else 0 for x in dfx['ARR_DELAY']]

dfx.describe()

#select the facts of interest for the purpose of the project
#EXTRACT SOME 9 important FEATURES To WORK WITH
#==============================================================================
# print(list(fdf.columns))
# ['DAY_OF_WEEK', 'UNIQUE_CARRIER', 'ORIGIN', 'DEST', 'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'DISTANCE', 'ARR_DELAY']
#==============================================================================
fdf = dfx.iloc[:, [2,3,9,14,16,18,19,24,21]]
fdf.isnull().sum() #Check ifthere exist null values in the data set

          #SOME STATISTICS FROM THE DATA SET
(fdf["DEST"]).value_counts()
(fdf["ORIGIN"]).value_counts()  
(fdf["UNIQUE_CARRIER"]).value_counts()
(fdf["DAY_OF_WEEK"]).value_counts()

         #SOME VISUALIZATIONS FROM THE DATA SET
sns.countplot(x="ARR_DELAY", data=fdf)
sns.countplot(x="UNIQUE_CARRIER", data=fdf) #Showing the airline with the most flights
sns.countplot(x="DAY_OF_WEEK", data=fdf) #Flight by day of the week graph       
sns.countplot(x="DEST", data=fdf) #Showing the Destination with the most flights
sns.countplot(x="ORIGIN", data=fdf) #Showing the ORIGIN AIRPORT with the most flights
         
#Examine the shape of the data set
print(fdf.shape) 

#Looking at the correlation between numeric attributes  of the dataset
sns.heatmap(fdf.corr())

#wHAT NEXT??
# Separate predictors from outcome
x =  fdf.iloc[:, [0,1,2,3,4,5,6,7]].values
y =  fdf.iloc[:,8].values

             #ENCODING ALL CATEGORICAL VARIABLES
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:,0])
x[:, 1] = labelencoder_x.fit_transform(x[:,1])
x[:, 2] = labelencoder_x.fit_transform(x[:,2])
x[:, 3] = labelencoder_x.fit_transform(x[:,3])
onehotencoder = OneHotEncoder(categorical_features = [0,1,2,3])
x = onehotencoder.fit_transform(x).toarray()
            
#SPLIT THE DATASET INTO TRAINING AND TEST SET WITH THE TEST SIZE AS 20% OF THE ORIGINAL DATA
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=0)

#Feature Scaling the training dataset ::: Standardization
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

#Using the Logistic Regression Algorithm to Train DataSet
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)
# classifier.coef_

#predicting the test set results using the Logistic Regression Model built
x_test = scaler.transform(x_test)
y_pred = classifier.predict(x_test)

#The Confusion Matrix As a measure of the prediction performance
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(x_test, y_test)))

#Apply the K-Fold Cross Validation to test the predictive power of your model
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
print("10-fold cross validation average accuracy: %.2f" % (accuracies.mean()))

#Compute precision, recall, F-measure and support
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

