import numpy as np
import pandas as pd
import csv
from sklearn.datasets import make_blobs
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_digits

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import MaxPooling1D as Kerasmaxpooling1d
from tensorflow.keras.layers import Conv1D as Kerasconv1d
from tensorflow.python.framework.ops import Tensor as Kerastensor

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import string
import re
import sys, pickle

#=========================
# IMPORTING MNIST DATA
#=========================

f = open('./mnist.pkl', 'rb')
if sys.version_info < (3,):
    data = pickle.load(f)
else:
    data = pickle.load(f, encoding='bytes')
f.close()
print(np.shape(data))
print(type(data))
#=========================
# PREPROCESSING MNIST DATA
#=========================

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = data

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

data = (x_train, y_train), (x_test, y_test)
print(y_train[0:4])
print(np.shape(data))
print(type(data))

# z=np.transpose([y])
# a=np.concatenate((X,z),axis=1)

# with open('mnist.csv', 'w',newline='') as f:
#     writer = csv.writer(f, delimiter=',')
#     writer.writerows(a)


# #=========================
# # DATA PREPROCESSING
# #=========================

# batch_size = 32
# raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
#     "aclImdb/train",
#     batch_size=batch_size,
#     validation_split=0.2,
#     subset="training",
#     seed=1337,
# )
# raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
#     "aclImdb/train",
#     batch_size=batch_size,
#     validation_split=0.2,
#     subset="validation",
#     seed=1337,
# )
# raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
#     "aclImdb/test", batch_size=batch_size
# )

# # print(
# #     "Number of batches in raw_train_ds: %d"
# #     % tf.data.experimental.cardinality(raw_train_ds)
# # )
# # print(
# #     "Number of batches in raw_val_ds: %d" % tf.data.experimental.cardinality(raw_val_ds)
# # )
# # print(
# #     "Number of batches in raw_test_ds: %d"
# #     % tf.data.experimental.cardinality(raw_test_ds)
# # )

# # for text_batch, label_batch in raw_train_ds.take(1):
# #     for i in range(5):
# #         print(text_batch.numpy()[i])
# #         print(label_batch.numpy()[i])

# # Having looked at our data above, we see that the raw text contains HTML break
# # tags of the form '<br />'. These tags will not be removed by the default
# # standardizer (which doesn't strip HTML). Because of this, we will need to
# # create a custom standardization function.
# def custom_standardization(input_data):
#     lowercase = tf.strings.lower(input_data)
#     stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
#     return tf.strings.regex_replace(
#         stripped_html, "[%s]" % re.escape(string.punctuation), ""
#     )

# #================================
# # TEXT VECTORIZATION
# #================================

# # Model constants.
# max_features = 20000
# embedding_dim = 128
# sequence_length = 500

# # Now that we have our custom standardization, we can instantiate our text
# # vectorization layer. We are using this layer to normalize, split, and map
# # strings to integers, so we set our 'output_mode' to 'int'.
# # Note that we're using the default split function,
# # and the custom standardization defined above.
# # We also set an explicit maximum sequence length, since the CNNs later in our
# # model won't support ragged sequences.
# vectorize_layer = TextVectorization(
#     standardize=custom_standardization,
#     max_tokens=max_features,
#     output_mode="int",
#     output_sequence_length=sequence_length,
# )

# # Now that the vocab layer has been created, call `adapt` on a text-only
# # dataset to create the vocabulary. You don't have to batch, but for very large
# # datasets this means you're not keeping spare copies of the dataset in memory.

# # Let's make a text-only dataset (no labels):
# text_ds = raw_train_ds.map(lambda x, y: x)
# # Let's call `adapt`:
# vectorize_layer.adapt(text_ds)






#====================================================================================
# inputs = tf.keras.Input(shape=(None, None, 3))
# x = tf.constant([1., 2., 3., 4., 5.])
# inputs = tf.reshape(x, [1, 5, 1])
# print(type(inputs))
# print(inputs)

#inputs = tf.keras.Input(shape=(None, 3))
#print(type(inputs))
#x = Kerasconv1d(kernel_size=2, filters=32)(inputs)
#print(x)
#print(type(x))
#y = Kerasmaxpooling1d(pool_size=2)(x)
#print(type(y))
#print(y)
#a = pd.read_csv('dataset1.csv')
#print(a)

#=====================================================================
# X, _ = load_digits(return_X_y=True)
# print(np.shape(X))
# with open('digits.csv', 'w',newline='') as f:
#     writer = csv.writer(f, delimiter=',')
#     writer.writerows(X)

#=====================================================================
# rng = np.random.RandomState(1)
# X = rng.randint(5, size=(6, 10))
# y = np.array([1, 2, 3, 4, 5, 6])

# z=np.transpose([y])
# a=np.concatenate((X,z),axis=1)

# with open('dataset7.csv', 'w',newline='') as f:
#     writer = csv.writer(f, delimiter=',')
#     writer.writerows(a)

#=====================================================================
# n_samples = 300

# # generate random sample, two components
# np.random.seed(0)

# # generate spherical data centered on (20, 20)
# shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])

# # generate zero centered stretched Gaussian data
# C = np.array([[0., -0.7], [3.5, .7]])
# stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)

# # concatenate the two datasets into the final training set
# X_train = np.vstack([shifted_gaussian, stretched_gaussian])

# with open('dataset6.csv', 'w',newline='') as f:
#     writer = csv.writer(f, delimiter=',')
#     writer.writerows(X_train)

#=====================================================================
# # Generate sample data
# np.random.seed(0)
# X = np.sort(5 * np.random.rand(40, 1), axis=0)
# y = np.sin(X).ravel()

# # Add noise to targets
# y[::5] += 1 * (0.5 - np.random.rand(8))

# z=np.transpose([y])
# a=np.concatenate((X,z),axis=1)

# with open('dataset2.csv', 'w',newline='') as f:
#     writer = csv.writer(f, delimiter=',')
#     writer.writerows(a)

# #=====================================================================
# # Generate data
# X, y = make_blobs(n_samples=1000, random_state=42, cluster_std=5.0)
# X_train, y_train = X[:600], y[:600]
# X_valid, y_valid = X[600:800], y[600:800]
# X_train_valid, y_train_valid = X[:800], y[:800]
# X_test, y_test = X[800:], y[800:]

# z = np.transpose([y_train_valid])
# a=np.concatenate((X_train_valid,z),axis=1)

# with open('dataset4.csv', 'w',newline='') as f:
#     writer = csv.writer(f, delimiter=',')
#     writer.writerows(a)
#=====================================================================
# # Load diabetes data
# X, y = load_diabetes(return_X_y=True)
# z = np.transpose([y])
# a=np.concatenate((X,z),axis=1)

# print(np.shape(X))

# with open('diabetes.csv', 'w',newline='') as f:
#     writer = csv.writer(f, delimiter=',')
#     writer.writerows(a)
#=====================================================================
#Generate data
# X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)
# z = np.transpose([y])
# a=np.concatenate((X,z),axis=1)
# with open('dataset5.csv', 'w',newline='') as f:
#     writer = csv.writer(f, delimiter=',')
#     writer.writerows(a)
