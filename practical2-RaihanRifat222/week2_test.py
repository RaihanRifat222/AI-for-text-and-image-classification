import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import keras
import unittest

import week2

class TestMNIST(unittest.TestCase):
    def test_summary_mnist(self):
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        result = week2.summary_mnist(train_labels, test_labels, range(10))
        self.assertDictEqual(result, 
                             {'train_labels_counts': [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], 
                              'test_labels_counts': [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]})
 
    def test_build_mnist_model(self):
        self.maxDiff = None
        model = week2.build_mnist_model(512,'relu',0.5)
        self.assertTrue(isinstance(model.layers[0], keras.layers.Dense))
        self.assertEqual(model.layers[0].output.shape, (None, 512))
        self.assertTrue(isinstance(model.layers[1], keras.layers.Dropout))
        self.assertTrue(isinstance(model.layers[2], keras.layers.Dense))
        self.assertEqual(model.layers[2].output.shape, (None, 10))

        model = week2.build_mnist_model(256,'sigmoid',0)
        self.assertTrue(isinstance(model.layers[0], keras.layers.Dense))
        self.assertEqual(model.layers[0].output.shape, (None, 256))
        self.assertTrue(isinstance(model.layers[1], keras.layers.Dense))
        self.assertEqual(model.layers[1].output.shape, (None, 10))

class TestFlowers(unittest.TestCase):
    def test_build_flowers_model(self):
        self.MaxDiff = None
        model = week2.build_flowers_model(1, 128, 0.7)
        self.assertTrue(isinstance(model.layers[0], keras.layers.Flatten))
        self.assertEqual(model.layers[0].output.shape, (None, 150528))
        self.assertTrue(isinstance(model.layers[1], keras.layers.Dense))
        self.assertEqual(model.layers[1].output.shape, (None, 128))
        self.assertTrue(isinstance(model.layers[2], keras.layers.Dropout))
        self.assertTrue(isinstance(model.layers[3], keras.layers.Dense))
        self.assertEqual(model.layers[3].output.shape, (None, 6))

        model = week2.build_flowers_model(2, 128, 0)
        self.assertTrue(isinstance(model.layers[0], keras.layers.Flatten))
        self.assertEqual(model.layers[0].output.shape, (None, 150528))
        self.assertTrue(isinstance(model.layers[1], keras.layers.Dense))
        self.assertEqual(model.layers[1].output.shape, (None, 128))
        self.assertTrue(isinstance(model.layers[2], keras.layers.Dense))
        self.assertEqual(model.layers[2].output.shape, (None, 128))
        self.assertTrue(isinstance(model.layers[3], keras.layers.Dense))
        self.assertEqual(model.layers[3].output.shape, (None, 6))

if __name__ == "__main__":
    unittest.main()
