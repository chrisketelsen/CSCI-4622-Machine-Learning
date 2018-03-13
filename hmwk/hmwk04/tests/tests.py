import sys

testSuite = sys.argv[1]

import unittest
import logging as log
import numpy as np

class TestNN(unittest.TestCase):

    def setUp(self):
        self.nn = Network([2,3,2])
        self.nn.W[0] = np.array([[0.1, 0.2],[-0.1, -0.2], [-0.1, 0.3]])
        self.nn.W[1] = np.array([[0.3, -0.2, -0.1], [-0.1, 0.3, 0.1]])
        self.nn.b[0] = np.array([-0.1, 0.1, 0.3])
        self.nn.b[1] = np.array([-0.2, 0.1])
        self.X_train = np.array([[1.0, 2.0]])
        self.y_train = np.array([[1, 0]], dtype=int)

    def testForwardProp(self): 
        """
        test forward propagation 
        """
        self.nn.forward_prop(self.X_train[0])
        self.assertAlmostEqual(self.nn.a[-1][0], 0.45768803332214286)
        self.assertAlmostEqual(self.nn.a[-1][1], 0.55713001269439544)

    def testPredict(self): 
        """
        test forward propagation 
        """
        yhat = self.nn.predict(self.X_train)
        self.assertEqual(yhat[0][0], 0)
        self.assertEqual(yhat[0][1], 1)

    def testBackProp(self):
        """
        test back propagation 
        """
        self.nn.back_prop(self.X_train[0],self.y_train[0])

        self.assertAlmostEqual(self.nn.dW[0][0,0], -0.052474530494067473)
        self.assertAlmostEqual(self.nn.dW[0][0,1], -0.10494906098813495)
        self.assertAlmostEqual(self.nn.dW[0][1,0],  0.066216197205198363)
        self.assertAlmostEqual(self.nn.dW[0][1,1],  0.13243239441039673)
        self.assertAlmostEqual(self.nn.dW[0][2,0],  0.023518130014919033)
        self.assertAlmostEqual(self.nn.dW[0][2,1],  0.047036260029838066)

        self.assertAlmostEqual(self.nn.dW[1][0,0], -0.3246754823813483)
        self.assertAlmostEqual(self.nn.dW[1][0,1], -0.21763648429650878)
        self.assertAlmostEqual(self.nn.dW[1][0,2], -0.37418141781784953)
        self.assertAlmostEqual(self.nn.dW[1][1,0],  0.33354686367842828)
        self.assertAlmostEqual(self.nn.dW[1][1,1],  0.22358314901596715)
        self.assertAlmostEqual(self.nn.dW[1][1,2],  0.38440549142943564)

        self.assertAlmostEqual(self.nn.db[0][0], -0.052474530494067473)
        self.assertAlmostEqual(self.nn.db[0][1],  0.066216197205198363)
        self.assertAlmostEqual(self.nn.db[0][2],  0.023518130014919033)

        self.assertAlmostEqual(self.nn.db[1][0], -0.54231196667785708)
        self.assertAlmostEqual(self.nn.db[1][1],  0.55713001269439544)

    def testSGD(self):
        """
        test unregularized stochastic gradient descent 
        """
        self.nn.train(self.X_train,self.y_train, eta=0.25, lam=0.0, num_epochs=2, isPrint=False)

        self.assertAlmostEqual(self.nn.W[0][0,0],  0.12900460960671509)
        self.assertAlmostEqual(self.nn.W[0][0,1],  0.25800921921343017)
        self.assertAlmostEqual(self.nn.W[0][1,0], -0.1275584298240941)
        self.assertAlmostEqual(self.nn.W[0][1,1], -0.2551168596481882) 
        self.assertAlmostEqual(self.nn.W[0][2,0], -0.10614724024199987)
        self.assertAlmostEqual(self.nn.W[0][2,1],  0.28770551951600032)

        self.assertAlmostEqual(self.nn.W[1][0,0],  0.45404965242391082)
        self.assertAlmostEqual(self.nn.W[1][0,1], -0.10100861856180195)
        self.assertAlmostEqual(self.nn.W[1][0,2],  0.074090776885554571)
        self.assertAlmostEqual(self.nn.W[1][1,0], -0.25834286364088743)
        self.assertAlmostEqual(self.nn.W[1][1,1],  0.19825242013683891)
        self.assertAlmostEqual(self.nn.W[1][1,2], -0.078940420827764327)

        self.assertAlmostEqual(self.nn.b[0][0], -0.070995390393284924)
        self.assertAlmostEqual(self.nn.b[0][1],  0.072441570175905898)
        self.assertAlmostEqual(self.nn.b[0][2],  0.29385275975800013)

        self.assertAlmostEqual(self.nn.b[1][0],  0.053614322995520436)
        self.assertAlmostEqual(self.nn.b[1][1], -0.16068005771911104)

    def testRegularizedSGD(self):
        """
        test regularized stochastic gradient descent 
        """
        self.nn.train(self.X_train,self.y_train, eta=0.25, lam=1.0, num_epochs=2, isPrint=False)

        self.assertAlmostEqual(self.nn.W[0][0,0],  0.07956554258404433)
        self.assertAlmostEqual(self.nn.W[0][0,1],  0.15913108516808866)
        self.assertAlmostEqual(self.nn.W[0][1,0], -0.076329338984104794)
        self.assertAlmostEqual(self.nn.W[0][1,1], -0.15265867796820959)
        self.assertAlmostEqual(self.nn.W[0][2,0], -0.059585510408211258)
        self.assertAlmostEqual(self.nn.W[0][2,1],  0.1620789791835775)

        self.assertAlmostEqual(self.nn.W[1][0,0],  0.29981338807709029)
        self.assertAlmostEqual(self.nn.W[1][0,1], -0.02301230001884183)
        self.assertAlmostEqual(self.nn.W[1][0,2],  0.092120504443546303)
        self.assertAlmostEqual(self.nn.W[1][1,0], -0.18939477160834683)
        self.assertAlmostEqual(self.nn.W[1][1,1],  0.077857379836854229)
        self.assertAlmostEqual(self.nn.W[1][1,2], -0.094503309481706793)

        self.assertAlmostEqual(self.nn.b[0][0], -0.073404799260076481)
        self.assertAlmostEqual(self.nn.b[0][1],  0.075782148690570317)
        self.assertAlmostEqual(self.nn.b[0][2],  0.29519460646585632)

        self.assertAlmostEqual(self.nn.b[1][0],  0.055041238882961835)
        self.assertAlmostEqual(self.nn.b[1][1], -0.15945721790381118)



if testSuite == "prob 3A":
    prob3A = unittest.TestSuite()
    for test in ["testForwardProp"]:
        prob3A.addTest(TestNN(test))
    runner = unittest.TextTestRunner(verbosity=2).run(prob3A)

if testSuite == "prob 3B":
    prob3B = unittest.TestSuite()
    for test in ["testPredict"]:
        prob3B.addTest(TestNN(test))
    runner = unittest.TextTestRunner(verbosity=2).run(prob3B)

if testSuite == "prob 3C":
    prob3C = unittest.TestSuite()
    for test in ["testBackProp"]:
        prob3C.addTest(TestNN(test))
    runner = unittest.TextTestRunner(verbosity=2).run(prob3C)

if testSuite == "prob 3D":
    prob3D = unittest.TestSuite()
    for test in ["testSGD"]:
        prob3D.addTest(TestNN(test))
    runner = unittest.TextTestRunner(verbosity=2).run(prob3D)

if testSuite == "prob 3E":
    prob3D = unittest.TestSuite()
    for test in ["testRegularizedSGD"]:
        prob3D.addTest(TestNN(test))
    runner = unittest.TextTestRunner(verbosity=2).run(prob3D)


