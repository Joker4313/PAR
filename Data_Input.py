import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def read_file(Path):
    """Read the .txt, .data or .csv file as dataframe. Input a parameter 'Path', which indicates the location of the file."""
    dataframe = pd.read_csv(filepath_or_buffer=Path, header=None, delim_whitespace=True)
    return dataframe


X_train = read_file("Data Set/UCI HAR Dataset/train/X_train.txt")
X_train_arr = np.array(X_train)
X_train.to_csv('X_train.csv')



def return_X_train_DF():
    """Return dataframe of 'X_train'"""
    return X_train


def return_X_train_Array():
    """Return np_array of 'X_train'"""
    return X_train_arr


# labels in 'int' format
y_train = read_file('Data Set/UCI HAR Dataset/train/y_train.txt').astype(int)
y_train_arr = np.array(y_train)
y_train.to_csv('y_train.csv')

def return_y_train_DF():
    """Return dataframe of 'y_train'"""
    return y_train


def return_y_train_Array():
    """Return np_array of 'y_train'"""
    return y_train_arr

X_test = read_file('Data Set/UCI HAR Dataset/test/X_test.txt').astype(float)
X_test_arr = np.array(X_test)
X_test.to_csv('X_test.csv')

def return_X_test_DF():
    """Return dataframe of 'X_test'"""
    return X_test


def return_X_test_Array():
    """Return np_array of 'X_test'"""
    return X_test_arr


# labels in 'int' format
y_test = read_file('Data Set/UCI HAR Dataset/test/y_test.txt').astype(int)
y_test_arr = np.array(y_test)
y_test.to_csv('y_test.csv')

def return_y_test_DF():
    """Return dataframe of 'y_test'"""
    return y_test


def return_y_test_Array():
    """Return np_array of 'y_test'"""
    return y_test_arr


# print(X_train)
#print(y_train)