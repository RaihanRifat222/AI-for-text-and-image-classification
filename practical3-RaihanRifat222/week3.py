import numpy as np

from tensorflow.keras import models
from tensorflow.keras import layers


def build_mnist_conv_model(num_convolutions, num_filters, kernel_sizes, pool_sizes):
    """Return a Keras model that has a series of convolution+MaxPool layers, followed
    by a Flatten layer, and the final classification layer.
       - num_convolutions: The number of convolution+MaxPooling.
       - num_filters: A list that contains the number of filters in each convolution
       - kernel_sizes: A list that contains the kernel size of each convolution layer.
       - pool_sizes: A list that contains the pool sizes. If the value is 0, then there 
          is no MaxPooling layer.
    >>> model = build_mnist_conv_model(3, [32,64,128], [3,3,3], [2,2,0])
    >>> model.summary()
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 26, 26, 32)        320       
    <BLANKLINE>
     max_pooling2d (MaxPooling2  (None, 13, 13, 32)        0         
     D)                                                              
    <BLANKLINE>
     conv2d_1 (Conv2D)           (None, 11, 11, 64)        18496     
    <BLANKLINE>
     max_pooling2d_1 (MaxPoolin  (None, 5, 5, 64)          0         
     g2D)                                                            
    <BLANKLINE>
     conv2d_2 (Conv2D)           (None, 3, 3, 128)         73856     
    <BLANKLINE>
     flatten (Flatten)           (None, 1152)              0         
    <BLANKLINE>
     dense (Dense)               (None, 10)                11530     
    <BLANKLINE>
    =================================================================
    Total params: 104202 (407.04 KB)
    Trainable params: 104202 (407.04 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    """

    model = models.Sequential()
    for i in range (num_convolutions):
        if i == 0:
            model.add(layers.Conv2D(filters = num_filters[i], kernel_size = kernel_sizes[i] , activation = 'relu', input_shape = (28,28,1)))
        else:
            model.add(layers.Conv2D(filters = num_filters[i], kernel_size = kernel_sizes[i], activation = 'relu'))
        
        if pool_sizes[i]>0:
            model.add(layers.MaxPooling2D(pool_size = pool_sizes[i]))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation = 'softmax'))
        
        # if pool_sizes[i]:
        #     model.add()
    return model


def count_classification_errors(targets, predictions):
    """Return a Python dictionary that lists the counts of classification errors
    >>> targets = ['daisy','daisy','sunflower','tulip','daisy','sunflower']
    >>> predictions = ['daisy','sunflower','tulip','daisy','sunflower','tulip']
    >>> count_classification_errors(targets, predictions)
    {'daisy->sunflower': 2, 'sunflower->tulip': 2, 'tulip->daisy': 1}
    """
    from collections import Counter
    result = Counter()
    for i in range(len(targets)):
        if targets[i] != predictions[i]:
            result.update([targets[i] + '->' + predictions[i]])

    return dict(result)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
