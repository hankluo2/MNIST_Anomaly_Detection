import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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


def oversample_data2(x_train, y_train, bool_train_labels):
    # General image augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # 在数据集上将输入平均值设置为0
        samplewise_center=False,  # 将每个样本的平均值设置为0
        featurewise_std_normalization=False,  # 将输入除以数据集的std
        samplewise_std_normalization=False,  # 将每个输入除以它的std
        zca_whitening=False,  # 使用ZCA白化
        rotation_range=10,  # 在范围内随机旋转图像（0到180度）
        zoom_range=0.1,  # 随机缩放图像
        width_shift_range=0.1,  # 水平随机移动图像（总宽度的一部分）
        height_shift_range=0.1,  # 垂直随机移动图像（总高度的一部分）
        horizontal_flip=False,  # 随机翻转图像
        vertical_flip=False)  # 随机翻转图像

    # Select anomalous values
    ax_train = x_train[~bool_train_labels]
    ay_train = y_train[~bool_train_labels]
    # print(ax_train.shape, ay_train.shape)

    iterator = datagen.flow(ax_train, ay_train, batch_size=32)

    cnt = 0
    res_ax_train = []
    res_ay_train = []

    while cnt * 32 <= len(ax_train):
        batched_images = next(iterator)  # generate infinitively
        # print(batched_images[0].shape, batched_images[1].shape)
        for i in range(32):
            try:
                img = batched_images[0][i]
                label = batched_images[1][i]
                res_ax_train.append(img)
                res_ay_train.append(label)
            except:
                pass

        cnt += 1
    # print(cnt * 32)

    res_ax_train = np.array(res_ax_train, dtype=np.uint8)
    res_ay_train = np.array(res_ay_train, dtype=np.uint8)

    anomaly_features = np.concatenate([ax_train, res_ax_train], axis=0)  # add augmentation images
    anomaly_labels = np.concatenate([ay_train, res_ay_train], axis=0)
    # print(anomaly_features.shape, anomaly_labels.shape)

    pos_features = x_train[bool_train_labels]  # non-anomalies
    pos_labels = y_train[bool_train_labels]  # anomalies

    ids = np.arange(len(pos_features))
    choices = np.random.choice(ids, len(anomaly_features))

    res_pos_features = pos_features[choices]
    res_pos_labels = pos_labels[choices]

    resampled_features = np.concatenate([res_pos_features, anomaly_features], axis=0)
    resampled_labels = np.concatenate([res_pos_labels, anomaly_labels], axis=0)

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
