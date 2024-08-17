import keras
import unittest

import week3

class BasicTests(unittest.TestCase):
 
    def test_build_mnist_conv_model(self):
        self.maxDiff = None
        model = week3.build_mnist_conv_model(3, [32,64,128], [3,3,3], [2,2,0])
        target = """Model: "sequential"
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
        #print(get_model_summary(model))
        self.assertTrue(isinstance(model.layers[0], keras.layers.Conv2D))
        self.assertEqual(model.layers[0].output_shape, (None, 26, 26, 32))
        self.assertTrue(isinstance(model.layers[1], keras.layers.MaxPooling2D))
        self.assertEqual(model.layers[1].output_shape, (None, 13, 13, 32))
        self.assertTrue(isinstance(model.layers[2], keras.layers.Conv2D))
        self.assertEqual(model.layers[2].output_shape, (None, 11, 11, 64))
        self.assertTrue(isinstance(model.layers[3], keras.layers.MaxPooling2D))
        self.assertEqual(model.layers[3].output_shape, (None, 5, 5, 64))
        self.assertTrue(isinstance(model.layers[4], keras.layers.Conv2D))
        self.assertEqual(model.layers[4].output_shape, (None, 3, 3, 128))
        self.assertTrue(isinstance(model.layers[5], keras.layers.Flatten))
        self.assertTrue(isinstance(model.layers[6], keras.layers.Dense))
        self.assertEqual(model.layers[6].output_shape, (None, 10))

    def test_count_classification_errors(self):
        targets = ['daisy','daisy','sunflower','tulip','daisy','sunflower']
        predictions = ['daisy','sunflower','tulip','daisy','sunflower','tulip']
        targetResult = {'daisy->sunflower': 2, 'sunflower->tulip': 2, 'tulip->daisy': 1}
        counts = week3.count_classification_errors(targets, predictions)
        self.assertDictEqual(targetResult, counts)
        
if __name__ == "__main__":
    unittest.main()
