# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 13:03:23 2018

@author: hkkam
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#IMPORT THE DATASET
df = pd.read_csv('C:/Users/hkkam/Desktop/MLProject/air.csv')

#Visualize to check the correlation between the features
sns.heatmap(df.corr())

df1 = df.loc[(df['ORIGIN_CITY_NAME'] == 'Atlanta, GA') | (df['ORIGIN_CITY_NAME'] == 'Chicago, IL') | (df['ORIGIN_CITY_NAME'] == 'New York, NY') | (df['ORIGIN_CITY_NAME'] == 'Los Angeles, CA') | (df['ORIGIN_CITY_NAME'] == 'Dallas, TX') ]
#df1 = df1.reset_index(drop=True)

dfs = df1.loc[(df1['UNIQUE_CARRIER'] == 'AS') | (df1['UNIQUE_CARRIER'] == 'DL') | (df1['UNIQUE_CARRIER'] == 'VX') | (df1['UNIQUE_CARRIER'] == 'B6') | (df1['UNIQUE_CARRIER'] == 'HA') | (df1['UNIQUE_CARRIER'] == 'WN') | (df1['UNIQUE_CARRIER'] == 'OO') | (df1['UNIQUE_CARRIER'] == 'UA') | (df1['UNIQUE_CARRIER'] == 'AA') | (df1['UNIQUE_CARRIER'] == 'EV')]
#dfs = dfs.reset_index(drop=True)

dfx = dfs.loc[(dfs['DEST_CITY_NAME'] == 'Atlanta, GA') | (dfs['DEST_CITY_NAME'] == 'Chicago, IL') | (dfs['DEST_CITY_NAME'] == 'New York, NY') | (dfs['DEST_CITY_NAME'] == 'Los Angeles, CA') | (dfs['DEST_CITY_NAME'] == 'Dallas, TX') ]
dfx = dfx.reset_index(drop=True)


#Extract rows where flight was not cancelled. i.e CANCELLED == 0
dfx= dfx[dfx["CANCELLED"] == 0] #Create a data frame for the successful flights
dfx = dfx.reset_index(drop=True)

#early arrival = 0 and late =1
dfx['ARR_DELAY'] = dfx['ARR_DELAY'] = [0 if x < 0 else 1 for x in dfx['ARR_DELAY']]
#EXTRACT SOME 11 important FEATURES To WORK WITH
#==============================================================================
# print(list(fdf.columns))
# ['DAY_OF_WEEK', 'UNIQUE_CARRIER', 'ORIGIN', 'DEST', 'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'DISTANCE', 'ARR_DELAY']
#==============================================================================
fdf = dfx.iloc[:, [2,3,9,13,15,17,22,20]]
print(fdf.shape)
sns.heatmap(fdf.corr())
#(12906, 8)

#wHAT NEXT??
# Separate predictors from outcome
x =  fdf.iloc[:, [0,1,2,3,4,5,6,]].values
y = fdf.iloc[:,7].values
            
# ... and encode all categorical variables OF the training data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:,0])
x[:, 1] = labelencoder_x.fit_transform(x[:,1])
x[:, 2] = labelencoder_x.fit_transform(x[:,2])
x[:, 3] = labelencoder_x.fit_transform(x[:,3])
onehotencoder = OneHotEncoder(categorical_features = [0,1,2,3])
x = onehotencoder.fit_transform(x).toarray()

#SPLIT THE DATASET INTO TRAINING AND TEST SET MADE UP OF 0.2OF THE ORIGINAL DATA
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=0)

#Feature Scaling the training dataset ::: Standardization
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

#predicting the test result
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

#Apply the K-Fold Cross Validation to test the accuracy of your model
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
