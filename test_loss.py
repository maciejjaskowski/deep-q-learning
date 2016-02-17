import unittest
import theano.tensor as T
import theano
import numpy as np
import dqn


class MyTestCase(unittest.TestCase):
    def setUp(self):
        # data which will be used in various test methods
        self.avals = np.array([[1, 5, 3], [2, 4, 1]])
        self.bvals = np.array([[2, 3, 1, 8], [4, 2, 1, 1], [1, 4, 8, 5]])

    def test_validity(self):
        theano.config.on_unused_input = 'warn'
        a0_var = T.dmatrix('a0')
        r0_var = T.dmatrix('r0')
        fri_var = T.dmatrix("fri")
        out = T.dmatrix("out")
        out_stale = T.dmatrix("out_stale")

        f = theano.function([a0_var, r0_var, fri_var, out, out_stale],
                            dqn.build_loss(out, out_stale, a0_var, r0_var, fri_var, gamma=0.5))

        sqr_mean, mean, y, q = f(np.array([[1, 0, 0, 0, 0, 0],
                                           [0, 1, 0, 0, 0, 0]]),
                                 np.array([[1],
                                           [0]]),
                                 np.array([[1],
                                           [1]]),
                                 np.array([[-5, 1, 2, 3, 4, 7],
                                           [1, 4, 3, 4, 5, 9]]),
                                 np.array([[-5, 1, 2, 3, 4, 5],
                                           [1, 2, 3, 4, 5, 6]]))

        self.assertTrue(np.all(y == [[3.5], [3]]))
        self.assertTrue(np.all(q == [[-5], [4]]))
        self.assertTrue(sqr_mean == 36.625)
        self.assertTrue(mean == 3.75)

    def test_validity2(self):
        theano.config.on_unused_input = 'warn'
        a0_var = T.dmatrix('a0')
        r0_var = T.dmatrix('r0')
        fri_var = T.dmatrix("fri")
        out = T.dmatrix("out")
        out_stale = T.dmatrix("out_stale")

        f = theano.function([a0_var, r0_var, fri_var, out, out_stale],
                            dqn.build_loss(out, out_stale, a0_var, r0_var, fri_var, gamma=0.5))

        sqr_mean, mean, y, q = f(np.array([[1, 0, 0, 0, 0, 0],
                                           [0, 1, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 1]]),
                                 np.array([[1],
                                           [0],
                                           [5]]),
                                 np.array([[1],
                                           [1],
                                           [0]]),
                                 np.array([[-5, 1, 2, 3, 4, 7],
                                           [1, 4, 3, 4, 5, 9],
                                           [0, 9, 0, 3, 2, 1]]),
                                 np.array([[-5, 1, 2, 3, 4, 5],
                                           [1, 2, 3, 4, 5, 6],
                                           [8, 0, -1, -1, 2, 3]]))

        print(y, q)

