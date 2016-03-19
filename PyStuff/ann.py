#Import the packages
import numpy
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sknn.mlp import Regressor, Layer

#For logging
import logging
logging.basicConfig()

#Imputer is used to impute the NaN and infinity values
from sklearn.preprocessing import Imputer

#SVM Regression
from sklearn.svm import SVR
#Plotting
from matplotlib import pyplot as plt

#preprocessing
def remove_nan(input_dataset):
    '''Removes the NaN and inf in the dataframe in-place
    input: input_dataset, a panda dataframe
    '''
    print "No.of NaN values", input_dataset.isnull().sum()
    for column in input_dataset:
        #Replace NaN values with mean
        input_dataset[column].fillna(input_dataset[column].mean(), inplace=True)
    print "No.of NaN values", input_dataset.isnull().sum()
    #assert input_dataset.count(axis=1)
    return input_dataset

# Code starts here

#Creates a dataframe from the dataset(technical indicators)
input_dataset = pd.read_csv('nifty50_features.csv')
print type(input_dataset)
print "First few rows of the dataset:"
print input_dataset.head()
print "Dataset size:", input_dataset.shape

#Remove NaN
dataset = remove_nan(input_dataset)
print "Done with preprocessing\n*********************"

print "Loading input data:\n***********************************"
col = dataset.shape[1]

print "No.of columns", col

input_dataset = dataset.iloc[:,0:col-1]
#print input_dataset.shape[1]
#print input_dataset

print "Loading target data:\n***********************************"
target_dataset = dataset.iloc[:,col-1]
#print "Checking target_dataset", target_dataset

#Converting dataframes to numpy arrays
input_array = input_dataset.as_matrix()
output_array = target_dataset.as_matrix()
print type(input_array), type(output_array)

#Checking for infinity and NaN values in the input and output arrays

#print numpy.isnan(input_array).any()
#print numpy.isnan(output_array).any()
#print numpy.isinf(input_array).any()
#print numpy.isinf(output_array).any()
#First Model
#Random Forest Regression
####
###
##
#
print input_array.shape, output_array.shape
reg = RandomForestRegressor(n_estimators=10)
reg.fit(input_array, output_array)
print reg.predict([7876.94, 7876.94, 201.25, 96.0895023003,	89.5191739429,	97.5470278405,	0.4175298805, 3.9104976997	, 1.705969706, 41.29775641025662])


#Artifical Neural Network

ann_network = Regressor(layers= [Layer("Linear", units=10), Layer("Tanh", units=200), Layer("Linear", units=1)], learning_rate=0.0001, n_iter=10)

ann_network.fit(input_array, output_array)
print ann_network.predict([7876.94, 7876.94, 201.25, 96.0895023003,	89.5191739429, 97.5470278405, 0.4175298805, 3.9104976997, 1.705969706, 41.2977564])


#SVM Regression
svregressor = SVR(kernel='rbf', C=1e3, degree=3)
print input_array.shape, output_array.shape
svregressor.fit(input_array, output_array)
print svregressor.predict([7876.94, 7876.94, 201.25, 96.0895023003,	89.5191739429, 97.5470278405, 0.4175298805, 3.9104976997, 1.705969706, 41.2977564103])

#plt.plot(input_array)
#plt.show()
