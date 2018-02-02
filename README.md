# MLLRFlights
The essence of this machine learning project is to train a machine to predict with a good level of accuracy whether flights (popular US local Airlines) to and from popular destinations will be delayed or not on arrival. I would try to implement a solution to this problem using a binary classification method (logistic regression) to work.


Dataset available in MLP 2.0.rar folder

import libraries to be used for the project

#IMPORT THE DATASET THE THREE DATASET

#Join the 3  dataframes  together

#Check for Some Vital flight Summaries


#Extract ONLY flights which were succesful.   i.e Flights which were not cancelled

#Visualize to check the correlation between the features


#Extract the rows based on specific origin city name


#Extract the rows based on specific destination city name


#Extract the rows based on the top 10 airlines in the USA

#Reset the columns of the data frame


#Code the 'ARR_DELAY' column. Early arrival = 0 and late = 1


#select the facts of interest for the purpose of the project
#EXTRACT SOME 9 important FEATURES To WORK WITH
#['DAY_OF_WEEK', 'UNIQUE_CARRIER', 'ORIGIN', 'DEST', 'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'DISTANCE', 'ARR_DELAY']


#Check ifthere exist null values in the data set

 #SOME EXPLORATORY STATISTICS FROM THE DATA SET

#SOME VISUALIZATIONS FROM THE DATA SET

         
#Examine the shape of the data set


#Looking at the correlation between numeric attributes  of the dataset


#wHAT NEXT??
#Separate predictors from outcome


#ENCODING ALL CATEGORICAL VARIABLES
from sklearn.preprocessing import LabelEncoder , OneHotEncoder

            
#SPLIT THE DATASET INTO TRAINING AND TEST SET WITH THE TEST SIZE AS 20% OF THE ORIGINAL DATA

#Feature Scaling the training dataset ::: Standardization


#Using the Logistic Regression Algorithm to Train DataSet


#predicting the test set results using the Logistic Regression Model built

#The Confusion Matrix As a measure of the prediction performance

#Apply the K-Fold Cross Validation to test the avaerage predictive power of your model

#Compute precision, recall, F-measure and support


