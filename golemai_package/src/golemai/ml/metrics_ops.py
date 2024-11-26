import numpy as np


def true_positive(y_true, y_pred):
    return np.sum((y_true == 1) & (y_pred == 1))


def false_positive(y_true, y_pred):
    return np.sum((y_true == 0) & (y_pred == 1))


def false_negative(y_true, y_pred):
    return np.sum((y_true == 1) & (y_pred == 0))


def true_negative(y_true, y_pred):
    return np.sum((y_true == 0) & (y_pred == 0))
