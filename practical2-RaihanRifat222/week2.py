import numpy as np
from matplotlib import pyplot as plt

# Exercises related to MNST data set

from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers

def summary_mnist(train_labels, test_labels, labels_list):
    """Return a dictionary that counts the number of samples for each possible label
    >>> (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    >>> summary_mnist(train_labels, test_labels, range(10))
    {'train_labels_counts': [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], 'test_labels_counts': [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]}
    """
    from collections import Counter
    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)
    return {'train_labels_counts': [train_counts[n] for n in labels_list],
            'test_labels_counts': [test_counts[n] for n in labels_list]
    }

def build_mnist_model(hidden_size, hidden_activation, dropout_rate=0):
    """Return a Keras model that has 1 hidden layer based on these parameters:
      - hidden_size: Size of hidden layer
      - hidden_activation: Activation of the hidden layer
      - dropout_rate: Dropout rate of the dropout layer after the hidden layer.
          If the dropout rate is zero, then there should not be a dropout layer
    >>> model = build_mnist_model(512,'relu',0.5)
    >>> model.summary()
    Model: "sequential_2"\n\
    _________________________________________________________________\n\
     Layer (type)                Output Shape              Param #   \n\
    =================================================================\n\
     dense_5 (Dense)             (None, 512)               401920    \n\
    <BLANKLINE>\n\
     dropout_1 (Dropout)         (None, 512)               0         \n\
    <BLANKLINE>\n\
     dense_6 (Dense)             (None, 10)                5130      \n\
    <BLANKLINE>\n\
    =================================================================\n\
    Total params: 407,050\n\
    Trainable params: 407,050\n\
    Non-trainable params: 0\n\
    _________________________________________________________________
    >>> model = build_mnist_model(256,'sigmoid',0)
    >>> model.summary()
    Model: "sequential_3"\n\
    _________________________________________________________________\n\
     Layer (type)                Output Shape              Param #   \n\
    =================================================================\n\
     dense_7 (Dense)             (None, 256)               200960    \n\
    <BLANKLINE>\n\
     dense_8 (Dense)             (None, 10)                2570      \n\
    <BLANKLINE>\n\
    =================================================================\n\
    Total params: 203,530\n\
    Trainable params: 203,530\n\
    Non-trainable params: 0\n\
    _________________________________________________________________
"""

    from keras import models
    from keras import layers

    network = models.Sequential()
    network.add(layers.Dense(hidden_size, activation= hidden_activation, input_shape=(28 * 28,)))
  
    if dropout_rate:
        network.add(layers.Dropout(dropout_rate))
    network.add(layers.Dense(10, activation='softmax'))
    return network

# Exercises related to flowers data set

def build_flowers_model(num_hidden, hidden_size, dropout_rate=0):
    """Build a model for the flowers dataset that has 1 or more hidden layers 
    according to these parameters:
      - num_hidden: the number of hidden layers
      - hidden_size: the size of each hidden layer
      - dropout_rate: the dropout rate of each dropout layer. If the dropout rate 
          is zero, then there should not be any dropout layers
    All hidden layers should have a 'relu' activation function.
    >>> model = build_flowers_model(1, 128, 0.7)
    >>> model.summary()
    Model: "sequential"\n\
    _________________________________________________________________\n\
     Layer (type)                Output Shape              Param #   \n\
    =================================================================\n\
     flatten (Flatten)           (None, 150528)            0         \n\
    <BLANKLINE>\n\
     dense (Dense)               (None, 128)               19267712  \n\
    <BLANKLINE>\n\
     dropout (Dropout)           (None, 128)               0         \n\
    <BLANKLINE>\n\
     dense_1 (Dense)             (None, 6)                 774       \n\
    <BLANKLINE>\n\
    =================================================================\n\
    Total params: 19,268,486\n\
    Trainable params: 19,268,486\n\
    Non-trainable params: 0\n\
    _________________________________________________________________\n\
    >>> model = build_flowers_model(2, 128, 0)
    >>> model.summary()
    Model: "sequential_1"\n\
    _________________________________________________________________\n\
     Layer (type)                Output Shape              Param #   \n\
    =================================================================\n\
     flatten_1 (Flatten)         (None, 150528)            0         \n\
    <BLANKLINE>\n\
     dense_2 (Dense)             (None, 128)               19267712  \n\
    <BLANKLINE>\n\
     dense_3 (Dense)             (None, 128)               16512     \n\
    <BLANKLINE>\n\
     dense_4 (Dense)             (None, 6)                 774       \n\
    <BLANKLINE>\n\
    =================================================================\n\
    Total params: 19,284,998\n\
    Trainable params: 19,284,998\n\
    Non-trainable params: 0\n\
    _________________________________________________________________"""

    from keras import models
    from keras import layers

    model = models.Sequential()
    model.add(layers.Flatten(input_shape= (224, 224, 3)))
    for i in range (num_hidden):
        model.add(layers.Dense(hidden_size, activation = 'relu'))

        if dropout_rate:
            model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(6, activation = 'softmax'))
    return model

if __name__ == "__main__":
    import doctest
    doctest.testmod()
