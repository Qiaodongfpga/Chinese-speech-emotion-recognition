from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import matplotlib.pyplot as plt
import numpy as np

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

from tensorflow.keras import layers

batch_size = 128   # 大小可变，一次处理多少个样本
# Each MNIST image batch is a tensor of shape (batch_size, 28, 28).
# Each input sequence will be of size (28, 28) (height is treated like time).
input_dim = 28  # 必须和特征的列数目相等

units = 64   #  LSTM 中
output_size = 10  # labels are from 0 to 9

def build_model(allow_cudnn_kernel=True):
    # CuDNN is only available at the layer level, and not at the cell level.
    # This means `LSTM(units)` will use the CuDNN kernel,
    # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
    if allow_cudnn_kernel:
        # The LSTM layer with default options uses CuDNN.
        lstm_layer1 = tf.keras.layers.LSTM(units, input_shape=(None, input_dim))
        # lstm_layer2 = tf.keras.layers.LSTM(lstm_layer1)

    else:
        # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
        lstm_layer = tf.keras.layers.RNN(
            tf.keras.layers.LSTMCell(units),
            input_shape=(None, input_dim))

    model = tf.keras.models.Sequential([

        lstm_layer1,
        # lstm_layer2,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(output_size, activation='softmax')]

model = build_model(allow_cudnn_kernel=True)
model.summary()

    mnist = tf.keras.datasets.mnist

            (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    sample, sample_label = x_train[0], y_train[0]

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        batch_size=batch_size,
                        epochs=5)

