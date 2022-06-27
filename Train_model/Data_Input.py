import numpy as np
import pandas as pd


def read_file(path):
    """Read the .txt, .data or .csv file as dataframe. Input a parameter 'Path', which indicates the location of the
    file. """
    dataframe = pd.read_csv(filepath_or_buffer=path, header=None, delim_whitespace=True)
    return dataframe


x_train = read_file("../Data Set/UCI HAR Dataset/train/x_train.txt")
x_train_arr = np.array(x_train)
x_train.to_csv('x_train.csv')


def return_x_train_df():
    """Return dataframe of 'x_train'"""
    return x_train


def return_x_train_array():
    """Return np_array of 'x_train'"""
    return x_train_arr


# labels in 'int' format
y_train = read_file('../Data Set/UCI HAR Dataset/train/y_train.txt').astype(int)
y_train_arr = np.array(y_train)
y_train.to_csv('y_train.csv')


def return_y_train_df():
    """Return dataframe of 'y_train'"""
    return y_train


def return_y_train_array():
    """Return np_array of 'y_train'"""
    return y_train_arr


x_test = read_file('../Data Set/UCI HAR Dataset/test/x_test.txt').astype(float)
x_test_arr = np.array(x_test)
x_test.to_csv('x_test.csv')


def return_x_test_df():
    """Return dataframe of 'x_test'"""
    return x_test


def return_x_test_array():
    """Return np_array of 'x_test'"""
    return x_test_arr


# labels in 'int' format
y_test = read_file('../Data Set/UCI HAR Dataset/test/y_test.txt').astype(int)
y_test_arr = np.array(y_test)
y_test.to_csv('y_test.csv')


def return_y_test_df():
    """Return dataframe of 'y_test'"""
    return y_test


def return_y_test_array():
    """Return np_array of 'y_test'"""
    return y_test_arr

# print(x_train)
# print(y_train)
