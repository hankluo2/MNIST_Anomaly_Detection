from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

save_path = './result'
Path(save_path).mkdir(parents=True, exist_ok=True)


def plot_metrics(history, method):
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()
    # plt.show()
    plt.savefig(save_path + '/' + method + '.png')
    plt.close()


def plot_cm(labels, predictions, method, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    # plt.show()
    plt.savefig(save_path + '/' + method + '.png')
    with open(save_path + '/' + method + '_confusion_matrix_result.txt', 'w') as f:
        f.write('Legitimate Transactions Detected (True Negatives): ' + str(cm[0][0]) + '\n')
        f.write('Legitimate Transactions Incorrectly Detected (False Positives): ' + str(cm[0][1]) + '\n')
        f.write('Fraudulent Transactions Missed (False Negatives): ' + str(cm[1][0]) + '\n')
        f.write('Fraudulent Transactions Detected (True Positives): ' + str(cm[1][1]) + '\n')
        f.write('Total Fraudulent Transactions: ' + str(np.sum(cm[1])) + '\n')
    plt.close()
