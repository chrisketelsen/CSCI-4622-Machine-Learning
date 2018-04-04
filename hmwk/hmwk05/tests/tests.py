import sys

testSuite = sys.argv[1]

import unittest
import logging as log
import numpy as np

# =========================================================================
# It's been 0 days since these unit tests have been modified 
# =========================================================================

class TestNB(unittest.TestCase):

    def setUp(self):

        self.text_train = np.array(["work buy  money", "nigeria opportunity viagra", "fly   buy nigeria", "money buy fly", "fly home nigeria"])
        self.y_train = np.array([0, 1, 1, 1, 0], dtype=int)

    def testVocab(self): 
        """
        test vocabulary 
        """
        self.nb = TextNB(self.text_train, self.y_train)
        self.nb.train()
        self.assertEqual(set(self.nb.vocab.keys()), set(["work", "buy", "money", "nigeria", "opportunity", "viagra", "fly", "home"]))
        self.assertEqual(set(self.nb.vocab.values()), set(list(range(8))))

    def testClassCounts(self): 
        """
        test counts of documents in each class 
        """
        self.nb = TextNB(self.text_train, self.y_train)
        self.nb.train()
        self.assertEqual(self.nb.class_counts[0],2)
        self.assertEqual(self.nb.class_counts[1],3)

    def testFeatureCounts(self): 
        """
        test feature counts for each feature in each class 
        """
        self.nb = TextNB(self.text_train, self.y_train)
        self.nb.train()
        self.assertEqual(self.nb.feature_counts[0, self.nb.vocab["work"]], 1)
        self.assertEqual(self.nb.feature_counts[0, self.nb.vocab["buy"]], 1)
        self.assertEqual(self.nb.feature_counts[0, self.nb.vocab["money"]], 1)
        self.assertEqual(self.nb.feature_counts[0, self.nb.vocab["nigeria"]], 1)
        self.assertEqual(self.nb.feature_counts[0, self.nb.vocab["opportunity"]], 0)
        self.assertEqual(self.nb.feature_counts[0, self.nb.vocab["viagra"]], 0)
        self.assertEqual(self.nb.feature_counts[0, self.nb.vocab["fly"]], 1)
        self.assertEqual(self.nb.feature_counts[0, self.nb.vocab["home"]], 1)

        self.assertEqual(self.nb.feature_counts[1, self.nb.vocab["work"]], 0)
        self.assertEqual(self.nb.feature_counts[1, self.nb.vocab["buy"]], 2)
        self.assertEqual(self.nb.feature_counts[1, self.nb.vocab["money"]], 1)
        self.assertEqual(self.nb.feature_counts[1, self.nb.vocab["nigeria"]], 2)
        self.assertEqual(self.nb.feature_counts[1, self.nb.vocab["opportunity"]], 1)
        self.assertEqual(self.nb.feature_counts[1, self.nb.vocab["viagra"]], 1)
        self.assertEqual(self.nb.feature_counts[1, self.nb.vocab["fly"]], 2)
        self.assertEqual(self.nb.feature_counts[1, self.nb.vocab["home"]], 0)

    def testLogScore(self): 
        """
        test feature counts for each feature in each class 
        """
        self.nb = TextNB(self.text_train, self.y_train)
        self.nb.train() 

        class_scores = self.nb.predict_log_score(self.text_train[2])
        self.assertEqual(len(class_scores), 2)
        self.assertAlmostEqual(class_scores[0], -6.6850283075531438)
        self.assertAlmostEqual(class_scores[1], -5.763418954099742)

        class_scores = self.nb.predict_log_score(self.text_train[4])
        self.assertEqual(len(class_scores), 2)
        self.assertAlmostEqual(class_scores[0], -6.6850283075531438)
        self.assertAlmostEqual(class_scores[1], -6.862031242767852)

    def testPredict(self): 
        """
        test feature counts for each feature in each class 
        """
        self.nb = TextNB(self.text_train, self.y_train)
        self.nb.train() 

        yhat = self.nb.predict(self.text_train)

        self.assertEqual(len(yhat), 5)
        self.assertTrue(all(yh==yi for yh, yi in zip(yhat, self.y_train)))


if testSuite == "prob 2A":
    prob2A = unittest.TestSuite()
    for test in ["testVocab", "testClassCounts", "testFeatureCounts"]:
        prob2A.addTest(TestNB(test))
    runner = unittest.TextTestRunner(verbosity=2).run(prob2A)

if testSuite == "prob 2B":
    prob2B = unittest.TestSuite()
    for test in ["testLogScore"]:
        prob2B.addTest(TestNB(test))
    runner = unittest.TextTestRunner(verbosity=2).run(prob2B)

if testSuite == "prob 2C":
    prob2C = unittest.TestSuite()
    for test in ["testPredict"]:
        prob2C.addTest(TestNB(test))
    runner = unittest.TextTestRunner(verbosity=2).run(prob2C)


