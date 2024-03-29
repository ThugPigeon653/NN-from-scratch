import unittest
import numpy as np
import math
import sys
import os
from ops import Op, SigOp, SumOp, MSEOp


class TestOps(unittest.TestCase):

    def test_op_output(self):
        op = Op(x_dim=2, y_dim=3)
        op.output = np.array([[1, 2], [3, 4], [5, 6]])
        self.assertTrue(np.array_equal(op.output, np.zeros((3, 2))))

    def test_sigop_forward_backward(self):
        sigop = SigOp(x_dim=2, y_dim=3)
        inputs = np.array([[1, 2],[3, 4],[5,6]])
        output = sigop.forward_propagate({}, inputs)
        self.assertTrue(sigop.output.shape==inputs.shape)
        backward = sigop.backward_propagate({})
        self.assertTrue(backward.shape==inputs.shape)

    def test_sumop_forward_backward(self):
        sumop = SumOp(x_dim=3, y_dim=2)
        weights = np.array([[1, 2, 3], [6, 4, 2]])
        x = np.array([[2, 1]])
        bias = 1
        output = sumop.forward_propagate({}, weights, x, bias)
        backward = sumop.backward_propagate()
        self.assertTrue(backward.shape==(3,2))

    def test_mseop_forward_backward(self):
        mseop = MSEOp(x_dim=2, y_dim=3)
        inputs = np.array([[1, 2], [3, 4], [5, 6]])
        expected = np.array([[2, 4], [6, 8], [10, 12]])
        output = mseop.forward_propagate({}, inputs, expected)
        self.assertTrue(np.array_equal(mseop.output, np.array([[1, 2], [3, 4], [5, 6]])))
        backward = mseop.backward_propagate()
        self.assertEqual(backward, 21)

if __name__ == '__main__':
    unittest.main()
