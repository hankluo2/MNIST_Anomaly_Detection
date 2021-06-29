import tensorflow as tf
from tensorflow import keras

import os
import time
import tempfile
from pathlib import Path
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python import training

from method import *

# Data loading
mnist = tf.keras.datasets.mnist

# Data preprocessing
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalization

# Set anomaly number: optional
anomaly_num = 1

# Reorganize labels: binary-classification
y_train = np.array([0 if item != anomaly_num else 1 for item in y_train], dtype=np.uint8)
y_test = np.array([0 if item != anomaly_num else 1 for item in y_test], dtype=np.uint8)

# Resize numpy arrays to tensorflow Tensors
train_images = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
test_images = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

# Tag bool labels
bool_train_labels = y_train == 0

# Class-weight params
neg, pos = np.bincount(y_train)
total = neg + pos


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='class_weights', help='Train method of MNIST anomaly detection')
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    if opt.method is None:
        raise 'Not method selected. Choose one of baseline/class_weights/oversampling attached to --method arg.'

    if opt.method == 'baseline':
        # Baseline
        baseline(train_images, y_train, test_images, y_test, epochs=30)  # Train #1

    if opt.method == 'oversampling':
        # Oversampling Method
        resampled_features, resampled_labels = oversample_data2(x_train=train_images,
                                                                y_train=y_train,
                                                                bool_train_labels=bool_train_labels)  # oversampling
        res_train_images = tf.reshape(
            resampled_features,
            (resampled_features.shape[0], resampled_features.shape[1], resampled_features.shape[2], 1))  # tensor reshaping

        oversampling(res_train_images, resampled_labels, test_images, y_test, epochs=30)  # Train #2

    # Weighted class Method
    if opt.method == 'class_weights':
        class_weights(pos, neg, total, train_images, y_train, test_images, y_test, epochs=30)  # Train #3
