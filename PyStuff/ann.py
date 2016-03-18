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
input_dataset = pd.read_csv('new_features.csv')
print type(input_dataset)
print "First few rows of the dataset:"
print input_dataset.head()
print "Dataset size:", input_dataset.shape

#Remove NaN
input_dataset = remove_nan(input_dataset)


#Creating input and target numpy arrays
'''input_array = numpy.genfromtxt('new_features.csv', delimiter=",", names=True)
input_array = Imputer(missing_values='NaN', strategy='most_frequent', axis=0).fit_transform(input_array)

target_array = numpy.genfromtxt('nifty50.csv', delimiter=",")
target_array =  target_array[:,1]

#print "Read " + str(len(input_array)) + " rows of data"
#assert len(target_array) == len(input_array)

print input_array.shape
print target_array.shape

print numpy.any(numpy.isnan(input_array))
print numpy.any(numpy.isinf(input_array))

#preprocessing
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

#Random Forest Regression

input_dataset = pd.read_csv('features.csv')
print input_dataset["SMA"]
reg = RandomForestRegressor(n_estimators=10)
#reg.fit(input_array, target_array)
#print reg.predict([7876.94, 7876.94, 201.25, 96.0895023003,	89.5191739429,	97.5470278405,	0.4175298805, 3.9104976997	, 1.705969706])


#Artifical Neural Network

ann_network = Regressor(layers= [Layer("Linear", units=10), Layer("Tanh", units=200), Layer("Linear", units=1)], learning_rate=0.0001, n_iter=10)

input_array[input_array == numpy.inf] = 0
input_array[input_array == numpy.nan] = 0
print numpy.any(numpy.isinf(input_array))
print numpy.any(numpy.isnan(input_array))

#ann_network.fit(input_array, target_array)


#print ann_network.predict([7876.94, 7876.94, 201.25, 96.0895023003,	89.5191739429,	97.5470278405,	0.4175298805, 3.9104976997	, 1.705969706])


#SVM Regression
svregressor = SVR(kernel='rbf', C=1e3, degree=3)
print input_array.shape, target_array.shape
svregressor.fit(input_array, target_array)
print svregressor.predict([7876.94, 7876.94, 201.25, 96.0895023003,	89.5191739429,	97.5470278405,	0.4175298805, 3.9104976997	, 1.705969706, 41.2977564103])

plt.plot(input_array)
#plt.show()
'''
