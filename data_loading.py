import numpy as np
from scipy.io import arff
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder


def import_data(X_train, y_train, X_test, y_test, yticklabels):
    """
    Load and preprocess train and test sets


    Parameters
    ----------
    dataset: string
        Name of the dataset


    Returns
    -------
    X_train: array
        Train set without labels

    y_train: array
        Labels of the train set encoded

    X_test: array
        Test set without labels

    y_test: array
        Labels of the test set encoded

    y_train_nonencoded: array
        Labels of the train set non-encoded

    y_test_nonencoded: array
        Labels of the test set non-encoded
    """

    # Transform to continuous labels
    y_train_nonencoded, y_test_nonencoded = y_train, y_test

    # Reshape data to match 2D convolution filters input shape
    X_train = np.reshape(
        np.array(X_train),
        (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1),
        order="C",
    )
    X_test = np.reshape(
        np.array(X_test),
        (X_test.shape[0], X_test.shape[1], X_test.shape[2], 1),
        order="C",
    )

    print("\nDataset" + " " + "loaded")
    print("Training set size: {0}".format(len(X_train)))
    print("Testing set size: {0}".format(len(X_test)))

    return X_train, y_train, X_test, y_test, y_train_nonencoded, y_test_nonencoded, yticklabels
