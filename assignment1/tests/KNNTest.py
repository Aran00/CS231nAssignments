from __future__ import absolute_import
import unittest
import numpy as np
from unittest import TestCase
from cs231n.classifiers.k_nearest_neighbor import KNearestNeighbor


class TestKNNClassifier(TestCase):

    def setUp(self):
        self.classifier = KNearestNeighbor()
        x_train = np.array([[1, 3], [2, 4], [3, 7], [2, 5], [3, 4]])
        y_train = np.array([1, 0, 1, 0, 1])
        self.classifier.train(x_train, y_train)

    def tearDown(self):
        pass

    def test_compute_distances(self):
        X = np.array([[3, 2], [5, 4]])
        dist2 = self.classifier.compute_distances_two_loops(X)
        dist1 = self.classifier.compute_distances_one_loop(X)
        dist0 = self.classifier.compute_distances_no_loops(X)
        self.assertEquals(np.array_equal(dist2, dist1), True)
        self.assertEquals(np.array_equal(dist0, dist1), True)
        labels = self.classifier.predict_labels(dist0, k=2)
        print labels

if __name__ == '__main__':
    unittest.main()
