import pandas as pd
import numpy as np
import os


# Function that reads features names from the provided file and returns it as a list. Var filepath should be a path
# to the file with data, var numberOfColumns is a number of columns present in this file.
# Params: String, Int
# Returns: List
def load_column_names(filepath, numberOfColumns):
    # open file to load header
    opened_file = open(filepath, 'r')

    # load feature names from lines starting with @attribute
    namesOfColumns = []
    for i in range(numberOfColumns + 4):
        line = opened_file.readline()
        if "attribute" in line:
            line = line.split()  # split line by whitespaces
            namesOfColumns.append(line[1])  # name of the feature is the second thing in the line

    return namesOfColumns


# Function that loads data from file provided in the param. Var filepath need to be a path to file with data. It
# uses DataFrame so it keeps labels for features names and number labels for every sample. returns two DataFrames -
# one with samples and all the features, the other one with class. row with index X in one DataFrame corresponds
# to row in the other DataFrame with the same index.
# Params: String
# Returns: (DataFrame, DataFrame)
def load_data_from_file(filepath):
    # load array with raw ML data without header (lines starting with @), each row represents one sample,
    # each column but last represents features, the last column represents class
    data = np.genfromtxt(filepath, comments='@', delimiter=',', dtype='str')
    numberOfColumns = data.shape[1]

    # load feature names from lines starting with @attribute
    namesOfColumns = load_column_names(filepath, numberOfColumns)

    # split feature names and class name, nameOfClassColumn takes the value of the last object in namesOfFeatures,
    # namesOfFeatures has 1 less value
    nameOfClassColumn = namesOfColumns.pop()
    namesOfFeatures = namesOfColumns

    # create a list with just class values for each case
    classes = []
    for item in data:
        classes.append([item[-1]])

    # create features array which has all columns but the last which was the class value
    features = np.delete(data, -1, axis=1)

    Y_classes = pd.DataFrame(classes, columns=[nameOfClassColumn])
    X_features = pd.DataFrame(features, columns=namesOfFeatures)

    return X_features, Y_classes


# Function that loads data from all the .dat files from data_dir and returns it as a list of pairs. Var data_dir should
# be a path to the dir with data ending with /.
# (DataFrame, DataFrame).
# Params: String
# Returns: List(DataFrame, DataFrame)
def load_data_from_files(data_dir):
    datasets = []

    filesInDataDir = os.listdir(data_dir)
    for file in filesInDataDir:
        if ".dat" in file:
            datasets.append(load_data_from_file(data_dir + file))

    return datasets
