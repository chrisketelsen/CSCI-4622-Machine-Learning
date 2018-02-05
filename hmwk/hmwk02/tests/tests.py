import sys

testSuite = sys.argv[1]

import unittest
import logging as log
import numpy as np

class TestUnweightedKNN(unittest.TestCase):

    def setUp(self):
        self.X_train = np.array([[2,5], [5,5], [1,4], [2,4], [4,4], [0,2], [3,2], [4,2], [5,2], [4,1], [2,0], [6,0]])
        self.y_train = np.array([-1, +1, -1, -1, -1, +1, +1, +1, +1, -1, -1, -1])
        self.x = np.array([0,3])
        self.X = np.array([[0,3], [4,3.5], [1,5]])

    def test1NNclassify(self): 
        """
        test 1NN
        """
        k1nn = KNN(self.X_train, self.y_train, K=1) 
        self.assertAlmostEqual(k1nn.classify(self.x), 1)

    def test2NNclassify(self): 
        """
        test 2NN. Checks tie-breaking. 
        """
        k2nn = KNN(self.X_train, self.y_train, K=2) 
        self.assertAlmostEqual(k2nn.classify(self.x), 1)

    def test3NNclassify(self): 
        """
        test 3NN
        """
        k3nn = KNN(self.X_train, self.y_train, K=3) 
        self.assertAlmostEqual(k3nn.classify(self.x), -1)

    def test3NNpredict(self): 
        """
        test 3NN prediction 
        """
        k3p = KNN(self.X_train, self.y_train, K=3) 
        yhat = k3p.predict(self.X)
        for yihat, yi in zip(yhat, [-1, 1, -1]): 
            self.assertAlmostEqual(yihat, yi)

class TestWeightedKNN(unittest.TestCase):

    def setUp(self):
        self.X_train = np.array([[0,3],[2,3],[3,3],[-1,2],[1,2],[3,2],[0,1],[.5,1], [.5,1], [2,1], [-1,-1], [0,-1], [3,-1]])
        self.y_train = np.array([1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1])
        self.x11 = np.array([1,1])
        self.x33 = np.array([3,3])
        self.X = np.array([[1,1], [3,3]])

    def test5NNclassify(self): 
        """
        test 5NN
        """
        k5nn = KNN(self.X_train, self.y_train, K=5, distance_weighted=True) 
        self.assertAlmostEqual(k5nn.classify(self.x11), -1)

    def test3NNclassify(self): 
        """
        test 3NN. Checks divide-by-zero issue. 
        """
        k3nn = KNN(self.X_train, self.y_train, K=3, distance_weighted=True) 
        self.assertAlmostEqual(k3nn.classify(self.x33), 1)

    def test5NNpredict(self): 
        """
        test 5NN prediction
        """
        k5p = KNN(self.X_train, self.y_train, K=5, distance_weighted=True) 
        yhat = k5p.predict(self.X)
        for yihat, yi in zip(yhat, [-1, 1]): 
            self.assertAlmostEqual(yihat, yi)

class TestPerceptron(unittest.TestCase):

    def TestNegMistake(self): 

        X_train = np.array([[1.,1.],[0.5,-1.],[-1.,-1.],[-1.,1.]])
        y_train = np.array([1, -1, -1, -1])

        # Test won't work if you change initial guess for w, b
        perc = Perceptron(X=X_train, y=y_train)
        self.assertAlmostEqual(perc.w[0], 1.0)
        self.assertAlmostEqual(perc.w[1], 0.0)
        self.assertAlmostEqual(perc.b, 0.0)

        # Check w, b after one epoch 
        perc.train()
        self.assertAlmostEqual(perc.w[0], 0.5)
        self.assertAlmostEqual(perc.w[1], 1.0)
        self.assertAlmostEqual(perc.b, -1.0)
        self.assertAlmostEqual(perc.num_mistakes, 1.0)


    def TestPosMistake(self): 

        X_train = np.array([[1.,1.],[1,-1.],[-.5,-1.],[-1.,1.]])
        y_train = np.array([1, 1, 1, -1])

        # Test won't work if you change initial guess for w, b
        perc = Perceptron(X=X_train, y=y_train)
        self.assertAlmostEqual(perc.w[0], 1.0)
        self.assertAlmostEqual(perc.w[1], 0.0)
        self.assertAlmostEqual(perc.b, 0.0)

        # Check w, b after one epoch 
        perc.train()
        self.assertAlmostEqual(perc.w[0], 0.5)
        self.assertAlmostEqual(perc.w[1], -1.0)
        self.assertAlmostEqual(perc.b, 1.0)
        self.assertAlmostEqual(perc.num_mistakes, 1.0)


if testSuite == "prob 2A":
    prob2A = unittest.TestSuite()
    for test in ["test1NNclassify", "test2NNclassify", "test3NNclassify", "test3NNpredict"]:
        prob2A.addTest(TestUnweightedKNN(test))
    runner = unittest.TextTestRunner(verbosity=2).run(prob2A)

if testSuite == "prob 2B":
    prob2B = unittest.TestSuite()
    for test in ["test5NNclassify", "test3NNclassify", "test5NNpredict"]:
        prob2B.addTest(TestWeightedKNN(test))
    runner = unittest.TextTestRunner(verbosity=2).run(prob2B)

if testSuite == "prob 4A":
    prob4A = unittest.TestSuite()
    for test in ["TestNegMistake", "TestPosMistake"]:
        prob4A.addTest(TestPerceptron(test))
    runner = unittest.TextTestRunner(verbosity=2).run(prob4A)




