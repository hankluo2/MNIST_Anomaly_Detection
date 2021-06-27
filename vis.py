import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
from sklearn import decomposition

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Set anomaly_num
anomaly_num = 2
new_y_train = np.array([0 if item != anomaly_num else 1 for item in y_train], dtype=np.uint8)

# Reconstruct TRAIN DATA
data = x_train.reshape((x_train.shape[0], -1))
labels = new_y_train.reshape((y_train.shape[0], 1)).T
labels = y_train.reshape((y_train.shape[0], 1)).T

# Standardize data
standardized_data = StandardScaler().fit_transform(data)

# Find co-variance matrix
sample_data = standardized_data
covar_matrix = np.matmul(sample_data.T, sample_data)

values, vectors = eigh(covar_matrix, eigvals=(782, 783))
vectors = vectors.T

# Projecting the original data sample on the plane
new_coordinates = np.matmul(vectors, sample_data.T)

# Appending label to the 2d projected data (vertical stack)
new_coordinates = np.vstack((new_coordinates, labels)).T

# Creating a new data frame for ploting the labeled points.
# dataframe = pd.DataFrame(data=new_coordinates, columns=("1st_principal", "2nd_principal", "label"))

# PCA process
pca = decomposition.PCA()
pca.n_components = 2
pca_data = pca.fit_transform(sample_data)

pca_data = np.vstack((pca_data.T, labels)).T

pca_df = pd.DataFrame(data=pca_data, columns=('1st_principal', '2nd_principal', 'label'))
sns.FacetGrid(pca_df, hue='label', size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.savefig(str(anomaly_num) + '.png')
