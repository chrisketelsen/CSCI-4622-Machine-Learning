import sys

testSuite = sys.argv[1]

import unittest
import logging as log
import numpy as np

class TestLogReg(unittest.TestCase):

    def setUp(self):
        self.X_train = np.array([[1,2,1], [1,1,5], [1,2,5], [1,3,5], [1,1,6], [1,2,6], [1,5,1], [1,6,1], [1,7,1], [1,6,2], [1,7,2], [1,5,5]], dtype=float)
        self.y_train =  np.array([1 if ii < 6 else 0 for ii in range(X.shape[0])], dtype=float)

    def testPosUnregUpdate(self): 
        """
        test update based on positive example 
        """
        unreg_pos = LogReg(np.array([self.X_train[2,:].copy()]), np.array([self.y_train[2]]))
        unreg_pos.beta = np.ones_like(unreg_pos.beta)
        unreg_pos.train(eta=0.1, lam=0.0, num_epochs=1)
        self.assertAlmostEqual(unreg_pos.beta[0], 1.0000670700260932)
        self.assertAlmostEqual(unreg_pos.beta[1], 1.0001341400521866)
        self.assertAlmostEqual(unreg_pos.beta[2], 1.0003353501304664)

    def testNegUnregUpdate(self): 
        """
        test update based on negative example 
        """
        unreg_neg = LogReg(np.array([self.X_train[9,:].copy()]), np.array([self.y_train[9]]))
        unreg_neg.beta = np.ones_like(unreg_neg.beta)
        unreg_neg.train(eta=0.2, lam=0.0, num_epochs=1)
        self.assertAlmostEqual(unreg_neg.beta[0],  0.60004935783039448)
        self.assertAlmostEqual(unreg_neg.beta[1], -1.3997038530176331)
        self.assertAlmostEqual(unreg_neg.beta[2],  0.20009871566078896)

    def testShuffelUnregUpdate(self): 
        """
        test that training examples are being shuffeled 
        """
        unreg_shuff = LogReg(np.array(self.X_train[:4,:].copy()), np.array(self.y_train[:4]))
        firsts = []
        for ii in range(20):
            unreg_shuff.beta = np.zeros_like(unreg_shuff.beta)
            unreg_shuff.train(eta=0.1, lam=0.0, num_epochs=2)
            firsts.append(unreg_shuff.beta[0])
        self.assertTrue(not np.all(firsts==firsts[0]))  

    def testPosRegUpdate(self): 
        """
        test regularized update based on positive example 
        """
        reg_pos = LogReg(np.array([self.X_train[2,:].copy()]), np.array([self.y_train[2]]))
        reg_pos.beta = np.ones_like(reg_pos.beta)
        reg_pos.train(eta=0.1, lam=0.1, num_epochs=1)
        self.assertAlmostEqual(reg_pos.beta[0], 1.0000670700260932)
        self.assertAlmostEqual(reg_pos.beta[1], 0.98013414005218658)
        self.assertAlmostEqual(reg_pos.beta[2], 0.98033535013046635)

    def testNegRegUpdate(self): 
        """
        test update based on negative example 
        """
        reg_neg = LogReg(np.array([self.X_train[9,:].copy()]), np.array([self.y_train[9]]))
        reg_neg.beta = np.ones_like(reg_neg.beta)
        reg_neg.train(eta=0.2, lam=0.1, num_epochs=1)
        self.assertAlmostEqual(reg_neg.beta[0],  0.60004935783039448)
        self.assertAlmostEqual(reg_neg.beta[1], -1.4397038530176332)
        self.assertAlmostEqual(reg_neg.beta[2],  0.16009871566078895)

    def testPredict(self):
        lr = LogReg(self.X_train, self.y_train)
        lr.beta = np.array([4.0, -1.0, 0.0])
        yhat = lr.predict(self.X_train)
        self.assertTrue(np.sum(yhat==self.y_train)==12.0)

    def testAccuracy(self):
        lr = LogReg(self.X_train, self.y_train)
        lr.beta = np.array([0, -3.0, 4.0])
        self.assertAlmostEqual(lr.accuracy(self.X_train, self.y_train), 10./12)


if testSuite == "prob 3A":
    prob3A = unittest.TestSuite()
    for test in ["testPosUnregUpdate", "testNegUnregUpdate", "testShuffelUnregUpdate"]:
        prob3A.addTest(TestLogReg(test))
    runner = unittest.TextTestRunner(verbosity=2).run(prob3A)

if testSuite == "prob 3B":
    prob3B = unittest.TestSuite()
    for test in ["testPosRegUpdate", "testNegRegUpdate"]:
        prob3B.addTest(TestLogReg(test))
    runner = unittest.TextTestRunner(verbosity=2).run(prob3B)

if testSuite == "prob 3C":
    prob3C = unittest.TestSuite()
    for test in ["testPredict"]:
        prob3C.addTest(TestLogReg(test))
    runner = unittest.TextTestRunner(verbosity=2).run(prob3C)

if testSuite == "prob 3D":
    prob3D = unittest.TestSuite()
    for test in ["testAccuracy"]:
        prob3D.addTest(TestLogReg(test))
    runner = unittest.TextTestRunner(verbosity=2).run(prob3D)

