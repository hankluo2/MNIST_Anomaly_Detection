import tensorflow as tf
from tensorflow import keras

import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import *
from plot import *

EPOCHS = 10
BATCH_SIZE = 512

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_prc',
                                                  verbose=1,
                                                  patience=10,
                                                  mode='max',
                                                  restore_best_weights=True)


def oversample_data(x_train, y_train, bool_train_labels):
    # For oversampling
    pos_features = x_train[bool_train_labels]  # non-anomaly
    neg_features = x_train[~bool_train_labels]  # anomaly

    pos_labels = y_train[bool_train_labels]
    neg_labels = y_train[~bool_train_labels]

    ids = np.arange(len(pos_features))
    choices = np.random.choice(ids, len(neg_features))

    res_pos_features = pos_features[choices]
    res_pos_labels = pos_labels[choices]

    resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
    resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)

    order = np.arange(len(resampled_labels))
    np.random.shuffle(order)

    resampled_features = resampled_features[order]
    resampled_labels = resampled_labels[order]
    print("Resampled dataset shapes: {}, {}".format(resampled_features.shape, resampled_labels.shape))
    return resampled_features, resampled_labels


def baseline(x_train, y_train, x_test, y_test, epochs):
    """ Baseline Training """
    # model = make_model()
    model = lenet5()
    model.summary()
    baseline_history = model.fit(x_train,
                                 y_train,
                                 batch_size=BATCH_SIZE,
                                 epochs=EPOCHS if epochs <= 0 else epochs,
                                 callbacks=[early_stopping],
                                 validation_data=(x_test, y_test))

    plot_metrics(baseline_history, 'baseline')

    # train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)
    test_predictions_baseline = model.predict(x_test, batch_size=BATCH_SIZE)
    baseline_results = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)

    with open('./result/basline.txt', 'w') as f:
        for name, value in zip(model.metrics_names, baseline_results):
            f.write(str(name) + ': ' + str(value) + '\n')

    plot_cm(y_test, test_predictions_baseline, 'baseline_cm')


def class_weights(pos, neg, total, x_train, y_train, x_test, y_test, epochs):
    """Class weights"""
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total))
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))

    weighted_model = lenet5()

    weighted_history = weighted_model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS if epochs <= 0 else epochs,
        callbacks=[early_stopping],
        validation_data=(x_test, y_test),
        # The class weights go here
        class_weight=class_weight)

    plot_metrics(weighted_history, 'class_weighted')

    # train_predictions_weighted = weighted_model.predict(train_images, batch_size=BATCH_SIZE)
    test_predictions_weighted = weighted_model.predict(x_test, batch_size=BATCH_SIZE)

    weighted_results = weighted_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
    with open('./result/class_weighted.txt', 'w') as f:
        for name, value in zip(weighted_model.metrics_names, weighted_results):
            f.write(str(name) + ': ' + str(value) + '\n')

    plot_cm(y_test, test_predictions_weighted, 'class_weighted_cm')


def oversampling(x_train, y_train, x_test, y_test, epochs):
    """Class weights: 0.5, 0.5"""

    class_weight = {0: 0.5, 1: 0.5}

    weighted_model = lenet5()

    weighted_history = weighted_model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS if epochs == 0 else epochs,
        callbacks=[early_stopping],
        validation_data=(x_test, y_test),
        # The class weights go here
        class_weight=class_weight)

    plot_metrics(weighted_history, 'oversampling')

    # train_predictions_weighted = weighted_model.predict(train_images, batch_size=BATCH_SIZE)
    test_predictions_weighted = weighted_model.predict(x_test, batch_size=BATCH_SIZE)

    weighted_results = weighted_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
    with open('./result/oversampling.txt', 'w') as f:
        for name, value in zip(weighted_model.metrics_names, weighted_results):
            f.write(str(name) + ': ' + str(value) + '\n')

    plot_cm(y_test, test_predictions_weighted, 'oversampling_cm')
