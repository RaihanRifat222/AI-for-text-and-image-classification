import numpy as np
import unittest

import week1

class TestWeek1(unittest.TestCase):
    def test_greyscale_rowscolumns(self):
        img = np.array([[   0, 255,   0],  
                        [ 110, 127, 140]])
        assert week1.greyscale_rowscolumns(img) == (2, 3)

    def test_greyscale_invert(self):
        img = np.array([[   0, 255,   0], 
                        [  50, 200,  50], 
                        [ 110, 127, 140]])
        target = np.array([[255,   0, 255],
                        [205,  55, 205],
                        [145, 128, 115]])
        np.testing.assert_array_equal(week1.greyscale_invert(img), target)
        
    def test_greyscale_highest_luminosity(self):
        img = np.array([[   0, 250,   0],  
                        [  50, 200,  50],  
                        [ 110, 127, 140]])
        assert week1.greyscale_highest_luminosity(img) == 250

    def test_greyscale_blackout(self):
        img = np.array([[   0, 255,   0],  
                        [  50, 200,  50],  
                        [ 110, 127, 140]])
        target = np.array([[  0,   0,   0],
                        [ 50,   0,  50],
                        [110, 127, 140]])
        np.testing.assert_array_equal(week1.greyscale_blackout(img, 200),
                                    target)

    def test_colour_rowscolumns(self):
        img = np.array([[[255,   0,   0], [  0, 255,   0], [  0,   0, 255]],
                        [[  0,   0,   0], [255, 255, 255], [127, 127, 127]]])
        assert week1.colour_rowscolumns(img) == (2, 3)

    def test_remove_red(self):
        img = np.array([[[255,   0,   0], [  0, 255,   0], [  0,   0, 255]],
                        [[  0,   0,   0], [255, 255, 255], [127, 127, 127]]])
        target = np.array([[[  0,   0,   0],
                            [  0, 255,   0],
                            [  0,   0, 255]],
                        [[  0,   0,   0],
                            [  0, 255, 255],
                            [  0, 127, 127]]])

        np.testing.assert_array_equal(week1.remove_red(img),
                                    target)

    def test_to_greyscale(self):
        img = np.array([[[255,   0,   0], [  0, 255,   0], [  0,   0, 255]],
                        [[  0,   0,   0], [255, 255, 255], [127, 127, 127]]])
        target = np.array([[ 85.,  85.,  85.],
                        [  0., 255., 127.]])

        np.testing.assert_array_equal(week1.to_greyscale(img),
                                    target)

    def test_mirror_image(self):
        img = np.array([[[255,   0,   0], [  0, 255,   0], [  0,   0, 255]],
                        [[  0,   0,   0], [255, 255, 255], [127, 127, 127]]])
        target = np.array([[[  0,   0, 255],
                            [  0, 255,   0],
                            [255,   0,   0]],
                        [[127, 127, 127],
                            [255, 255, 255],
                            [  0,   0,   0]]])
        np.testing.assert_array_equal(week1.mirror_image(img),
                                    target)

if __name__ == "__main__":
    unittest.main()
