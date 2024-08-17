import unittest

import week4

class BasicTests(unittest.TestCase):
 
    def test_call_counts(self):
        target = {'ship': 8, 'frog': 6, 'cat': 6, 'automobile': 6, 'deer': 6, 'bird': 9, 'airplane': 10, 'truck': 4, 'dog': 8, 'horse': 3}
        self.assertDictEqual(week4.call_counts('sample_cifar10'), target)

    def test_sample_images(self):
        self.maxDiff=None
        target = [('sample_cifar10/7_4207.png', 'horse'), ('sample_cifar10/7_4197.png', 'horse'), ('sample_cifar10/7_4202.png', 'horse')]
        self.assertListEqual(sorted(week4.sample_images('sample_cifar10', 'horse')), sorted(target))
        
if __name__ == "__main__":
    unittest.main()
