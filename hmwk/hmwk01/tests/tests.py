import unittest
import logging as log
import numpy as np

class TestRegressionData(unittest.TestCase):

    def setUp(self):
        self.X, self.y = np.column_stack((np.arange(10, dtype=float), np.arange(10,20, dtype=float))), np.arange(10, dtype=float)
        self.data = RegressionData(self.X, self.y, train_frac=0.7, random_state=1235)
        self.centerdata = RegressionData(self.X, self.y, train_frac=0.7, center=True, standardize=False, random_state=1235)
        self.standdata = RegressionData(self.X, self.y, train_frac=0.7, center=True, standardize=True, random_state=1235)

    def test_split(self): 
        """
        test correct split proportions 
        """
        all_set= all(v is not None for v in [self.data.X_train, self.data.y_train, self.data.X_valid, self.data.y_valid])
        self.assertTrue(all_set)
        if all_set: self.assertTrue(self.data.X_train.shape[0] == self.data.y_train.shape[0] == 7 == 10-self.data.X_valid.shape[0] == 10-self.data.y_valid.shape[0])

    def test_shuffle(self):
        """
        test that rows have been shuffled correctly 
        """
        all_set= all(v is not None for v in [self.data.X_train, self.data.y_train, self.data.X_valid, self.data.y_valid])
        self.assertTrue(all_set)
        if all_set: self.assertTrue(not np.all(np.concatenate((self.data.y_train, self.data.y_valid)) == np.arange(10)))
        if all_set: self.assertTrue(np.all(np.sort(np.concatenate((self.data.y_train, self.data.y_valid))) == np.arange(10)))
        if all_set: self.assertTrue(np.all([self.data.X_train[ii,0]==self.data.y_train[ii] for ii in range(self.data.X_train.shape[0])]))
        if all_set: self.assertTrue(np.all([self.data.X_valid[ii,0]==self.data.y_valid[ii] for ii in range(self.data.X_valid.shape[0])]))

    def test_train_center(self):
        """
        test that train features have mean 0 
        """
        all_set= all(v is not None for v in [self.centerdata.X_train])
        self.assertTrue(all_set)
        if all_set:
            for jj in range(self.centerdata.X_train.shape[1]):
                self.assertAlmostEqual(np.mean(self.centerdata.X_train[:,jj]), 0)

    def test_valid_center(self):
        """
        test that validation features have been been centered based on training set 
        """
        all_set= all(v is not None for v in [self.data.X_train, self.centerdata.X_train, self.data.X_valid, self.centerdata.y_valid])
        self.assertTrue(all_set)
        if all_set:
            for jj in range(self.centerdata.X_valid.shape[1]):
                mujj = np.mean(self.data.X_train[:,jj])
                self.assertAlmostEqual(np.linalg.norm(self.centerdata.X_valid[:,jj] - (self.data.X_valid[:,jj]-mujj)), 0)


    def test_train_standardize(self):
        """
        test that train features have mean 0 and norm 1 
        """
        all_set= all(v is not None for v in [self.standdata.X_train, self.standdata.X_valid])
        self.assertTrue(all_set)
        if all_set:
            for jj in range(self.standdata.X_train.shape[1]):
                self.assertAlmostEqual(np.mean(self.standdata.X_train[:,jj]), 0)
                self.assertAlmostEqual(np.std(self.standdata.X_train[:,jj]), 1)

    def test_valid_standardize(self):
        """
        test that validation features have been transformed according to training set 
        """
        all_set= all(v is not None for v in [self.data.X_train, self.standdata.X_train, self.data.X_valid, self.standdata.X_valid])
        self.assertTrue(all_set)
        if all_set:
            for jj in range(self.standdata.X_valid.shape[1]):
                mujj = np.mean(self.data.X_train[:,jj])
                sdjj = np.std(self.data.X_train[:,jj]-mujj)
                self.assertAlmostEqual(np.std(self.standdata.X_valid[:,jj] - (self.data.X_valid[:,jj]-mujj)/sdjj), 0)


    def test_new_standardize(self):
        """
        test that new data matrix is normalized based on training set 
        """
        all_set= all(v is not None for v in [self.data.X_train])
        self.assertTrue(all_set)
        if all_set:
            Xnew = np.random.rand(20,2)
            Xnn = self.standdata.transform(Xnew)
            for jj in range(Xnn.shape[1]):
                mujj = np.mean(self.data.X_train[:,jj])
                sdjj = np.std(self.data.X_train[:,jj]-mujj)
                self.assertAlmostEqual(np.std(Xnn[:,jj] - (Xnew[:,jj]-mujj)/sdjj), 0)


# print("Testing Part A ...")
partA = unittest.TestSuite()
for test in ["test_split", "test_shuffle"]:
    partA.addTest(TestRegressionData(test))
runner = unittest.TextTestRunner(verbosity=1).run(partA)

partB = unittest.TestSuite()
for test in ["test_train_center", "test_valid_center", "test_train_standardize", "test_valid_standardize"]:
    partB.addTest(TestRegressionData(test))
runner = unittest.TextTestRunner(verbosity=1).run(partB)

partC = unittest.TestSuite()
for test in ["test_new_standardize"]:
    partC.addTest(TestRegressionData(test))
runner = unittest.TextTestRunner(verbosity=1).run(partC)





